"""Tests for engine/storage.py, using an in-memory SQLite database."""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from engine.schemas import (
    CriteriaDirection,
    DoseRegimen,
    FormInput,
    GoalRow,
    GoalStatus,
    PlanFrame,
    PlanType,
    StructureClass,
)
from engine.storage import GoalResult, Patient, Plan, init_db, load_analysis_frame, save_case


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine():
    """A fresh in-memory database per test."""
    return init_db("sqlite://")


def _goal(**overrides) -> GoalRow:
    defaults = dict(
        goal_key="ptv_v95",
        priority=1,
        roi_raw="PTV_4000",
        roi="PTV",
        goal_text="PTV V95% >= 95%",
        goal_type="V95%",
        criteria=CriteriaDirection.AT_LEAST,
        acceptance_level=95.0,
        parameter_value=95.0,
        achieved_value=97.2,
        status=GoalStatus.PASS,
        evaluable=True,
        structure_class=StructureClass.TARGET,
    )
    defaults.update(overrides)
    return GoalRow(**defaults)


def _form(**overrides) -> FormInput:
    defaults = dict(
        hn="H001",
        pt_no="Pt1",
        dose_regimen=DoseRegimen.HYPO,
        rx_cgy=4000,
        fractions=15,
        sib_boost=False,
        tx_room="Room1",
        mp1="Alice",
        mp2="Bob",
        ro="Dr. Carter",
        time_manual=30.0,
        time_auto=10.0,
        time_automanual=15.0,
    )
    defaults.update(overrides)
    return FormInput(**defaults)


def _frame(plan_type: PlanType, goals=None, **overrides) -> PlanFrame:
    defaults = dict(
        plan_type=plan_type,
        planning_time_min=20.0,
        is_straight_pass=True,
        source_filename=f"case_{plan_type.value}.txt",
        file_sha256="deadbeef",
    )
    defaults.update(overrides)
    return PlanFrame(goals=goals if goals is not None else [_goal()], **defaults)


# --------------------------------------------------------------------------- #
# init_db
# --------------------------------------------------------------------------- #


def test_init_db_creates_all_tables(engine):
    table_names = set(engine.dialect.get_table_names(engine.connect()))
    assert table_names == {"patients", "plans", "goal_results", "goal_corrections", "analysis_runs"}


def test_init_db_is_idempotent(engine):
    # Calling it again on the same engine's URL should not error or wipe data.
    with Session(engine) as session:
        session.add(Patient(pt_no="Pt1", hn="H1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4000,
                             fractions=15, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C"))
        session.commit()

    init_db("sqlite://")  # a different in-memory DB, but exercises the "create if missing" path
    with Session(engine) as session:
        assert session.exec(select(Patient)).one().pt_no == "Pt1"


# --------------------------------------------------------------------------- #
# save_case
# --------------------------------------------------------------------------- #


def test_save_case_persists_patient_plan_and_goals(engine):
    form = _form()
    frame = _frame(PlanType.MANUAL)

    patient_id = save_case(engine, form, [frame])
    assert isinstance(patient_id, int)

    with Session(engine) as session:
        patient = session.get(Patient, patient_id)
        assert patient.pt_no == "Pt1"
        assert patient.hn == "H001"
        assert patient.dose_regimen == DoseRegimen.HYPO

        plans = session.exec(select(Plan).where(Plan.patient_id == patient_id)).all()
        assert len(plans) == 1
        assert plans[0].plan_type == PlanType.MANUAL

        goals = session.exec(select(GoalResult).where(GoalResult.plan_id == plans[0].id)).all()
        assert len(goals) == 1
        assert goals[0].goal_key == "ptv_v95"


def test_save_case_multiple_plans_one_transaction(engine):
    form = _form(pt_no="Pt2")
    frames = [_frame(PlanType.MANUAL), _frame(PlanType.AUTO), _frame(PlanType.AUTO_MANUAL)]

    patient_id = save_case(engine, form, frames)

    with Session(engine) as session:
        plans = session.exec(select(Plan).where(Plan.patient_id == patient_id)).all()
        assert {p.plan_type for p in plans} == {PlanType.MANUAL, PlanType.AUTO, PlanType.AUTO_MANUAL}


def test_plan_type_round_trips_as_exact_string(engine):
    """CLAUDE.md rule 5: the stored/retrieved value must be the literal
    string "Auto+Manual", not the Python enum member name."""
    form = _form(pt_no="Pt3")
    patient_id = save_case(engine, form, [_frame(PlanType.AUTO_MANUAL)])

    with Session(engine) as session:
        plan = session.exec(select(Plan).where(Plan.patient_id == patient_id)).one()
        assert plan.plan_type.value == "Auto+Manual"
        assert plan.plan_type == "Auto+Manual"  # str-Enum equality


def test_save_case_duplicate_pt_no_rolls_back_transaction(engine):
    save_case(engine, _form(pt_no="Pt1"), [_frame(PlanType.MANUAL)])

    with pytest.raises(IntegrityError):
        save_case(engine, _form(pt_no="Pt1", hn="H999"), [_frame(PlanType.AUTO)])

    with Session(engine) as session:
        # Only the first patient/plan exist; the failed second save_case left
        # no partial rows behind (the whole call is one transaction).
        assert len(session.exec(select(Patient)).all()) == 1
        assert len(session.exec(select(Plan)).all()) == 1
        assert len(session.exec(select(GoalResult)).all()) == 1


def test_save_case_with_no_plans_still_creates_patient(engine):
    patient_id = save_case(engine, _form(), [])
    with Session(engine) as session:
        assert session.get(Patient, patient_id) is not None
        assert session.exec(select(Plan)).all() == []


# --------------------------------------------------------------------------- #
# load_analysis_frame
# --------------------------------------------------------------------------- #


def test_load_analysis_frame_excludes_hn(engine):
    """CLAUDE.md rule 4: HN must never appear in analysis output."""
    save_case(engine, _form(hn="SECRET-HN-999"), [_frame(PlanType.MANUAL)])

    df = load_analysis_frame(engine)

    assert "hn" not in df.columns
    assert not df.astype(str).isin(["SECRET-HN-999"]).any().any()


def test_load_analysis_frame_one_row_per_goal(engine):
    goals_manual = [_goal(goal_key="g1"), _goal(goal_key="g2")]
    goals_auto = [_goal(goal_key="g1")]
    save_case(
        engine,
        _form(pt_no="Pt1"),
        [_frame(PlanType.MANUAL, goals=goals_manual), _frame(PlanType.AUTO, goals=goals_auto)],
    )

    df = load_analysis_frame(engine)

    assert len(df) == 3
    assert set(df["plan_type"]) == {"Manual", "Auto"}
    assert set(df.loc[df["plan_type"] == "Manual", "goal_key"]) == {"g1", "g2"}


def test_load_analysis_frame_joins_across_patients(engine):
    save_case(engine, _form(pt_no="Pt1"), [_frame(PlanType.MANUAL)])
    save_case(engine, _form(pt_no="Pt2", hn="H002"), [_frame(PlanType.AUTO)])

    df = load_analysis_frame(engine)

    assert set(df["pt_no"]) == {"Pt1", "Pt2"}
    assert len(df) == 2


def test_load_analysis_frame_columns_are_plain_strings_not_enums(engine):
    save_case(engine, _form(), [_frame(PlanType.AUTO_MANUAL)])
    df = load_analysis_frame(engine)

    row = df.iloc[0]
    assert row["plan_type"] == "Auto+Manual" and isinstance(row["plan_type"], str)
    assert row["dose_regimen"] == "Hypo" and isinstance(row["dose_regimen"], str)
    assert row["criteria"] == "AtLeast" and isinstance(row["criteria"], str)
    assert row["status"] == "PASS" and isinstance(row["status"], str)
    assert row["structure_class"] == "TARGET" and isinstance(row["structure_class"], str)


def test_load_analysis_frame_empty_db(engine):
    df = load_analysis_frame(engine)
    assert df.empty
    assert "hn" not in df.columns


# --------------------------------------------------------------------------- #
# FormInput validation
# --------------------------------------------------------------------------- #


def test_form_input_rejects_nonpositive_planning_time():
    with pytest.raises(ValidationError):
        _form(time_manual=0)
    with pytest.raises(ValidationError):
        _form(time_auto=-5)


def test_form_input_allows_missing_planning_times():
    form = _form(time_manual=None, time_auto=None, time_automanual=None)
    assert form.time_manual is None


def test_form_input_time_for_maps_plan_type_to_field():
    form = _form(time_manual=11.0, time_auto=22.0, time_automanual=33.0)
    assert form.time_for(PlanType.MANUAL) == 11.0
    assert form.time_for(PlanType.AUTO) == 22.0
    assert form.time_for(PlanType.AUTO_MANUAL) == 33.0


def test_form_input_rejects_empty_required_strings():
    with pytest.raises(ValidationError):
        _form(pt_no="")


def test_form_input_rejects_nonpositive_rx_cgy_or_fractions():
    with pytest.raises(ValidationError):
        _form(rx_cgy=0)
    with pytest.raises(ValidationError):
        _form(fractions=0)


# --------------------------------------------------------------------------- #
# GoalRow validation
# --------------------------------------------------------------------------- #


def test_goal_row_rejects_invalid_priority():
    with pytest.raises(ValidationError):
        _goal(priority=4)
    with pytest.raises(ValidationError):
        _goal(priority=0)


def test_goal_row_allows_none_priority():
    """None means "not yet known" (RayStation's no-priority sentinel, or a
    pending correction) — kept so it can be reviewed, not rejected."""
    goal = _goal(priority=None)
    assert goal.priority is None


def test_goal_row_rejects_invalid_criteria_value():
    with pytest.raises(ValidationError):
        GoalRow(
            goal_key="g", priority=1, roi_raw="PTV", roi="PTV", goal_text="x",
            goal_type="Dmax", criteria="NotADirection", acceptance_level=1.0,
            parameter_value=1.0, evaluable=True, structure_class=StructureClass.TARGET,
        )


def test_goal_row_allows_nullable_achieved_value_and_status():
    goal = _goal(achieved_value=None, status=None, evaluable=False)
    assert goal.achieved_value is None
    assert goal.status is None
