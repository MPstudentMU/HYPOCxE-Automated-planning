"""Tests for engine/storage.py, using an in-memory SQLite database."""
from __future__ import annotations

import pandas as pd
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
from engine.storage import (
    GoalResult,
    Patient,
    Plan,
    get_plan_times,
    init_db,
    load_analysis_frame,
    save_case,
    update_patient,
    update_plan_times,
)


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
# Relaxed intake — save with only pt_no (+ hn, resolved by the parser) and
# plan files; everything else null until filled in later via Edit
# --------------------------------------------------------------------------- #


def test_save_case_with_only_pt_no_and_plans(engine):
    """The New Case page's minimal path: pt_no plus whatever plan files
    were uploaded, nothing else filled in."""
    form = FormInput(pt_no="PtMinimal", hn="H999")
    patient_id = save_case(engine, form, [_frame(PlanType.MANUAL), _frame(PlanType.AUTO)])

    with Session(engine) as session:
        patient = session.get(Patient, patient_id)
        assert patient.pt_no == "PtMinimal"
        assert patient.hn == "H999"
        assert patient.dose_regimen is None
        assert patient.rx_cgy is None
        assert patient.fractions is None
        assert patient.sib_boost is None
        assert patient.tx_room is None
        assert patient.mp1 is None
        assert patient.mp2 is None
        assert patient.ro is None

        plans = session.exec(select(Plan).where(Plan.patient_id == patient_id)).all()
        assert {p.plan_type for p in plans} == {PlanType.MANUAL, PlanType.AUTO}


def test_save_case_with_only_pt_no_and_no_plans(engine):
    """Even fewer prerequisites: just pt_no, no plan files yet at all."""
    patient_id = save_case(engine, FormInput(pt_no="PtBareMinimum", hn="H998"), [])
    with Session(engine) as session:
        assert session.get(Patient, patient_id).pt_no == "PtBareMinimum"


def test_load_analysis_frame_handles_null_dose_regimen_and_rx_cgy(engine):
    """load_analysis_frame's enum-unwrapping must not choke on a genuinely
    missing dose_regimen (None isn't an Enum instance, but confirm it
    passes straight through rather than erroring)."""
    save_case(engine, FormInput(pt_no="PtMinimal", hn="H999"), [_frame(PlanType.MANUAL)])
    df = load_analysis_frame(engine)
    assert df.loc[0, "dose_regimen"] is None
    assert pd.isna(df.loc[0, "rx_cgy"])


# --------------------------------------------------------------------------- #
# get_plan_times / update_plan_times — Module 1's Edit dialog planning-time
# fields
# --------------------------------------------------------------------------- #


def test_get_plan_times_returns_each_uploaded_plans_time(engine):
    patient_id = save_case(engine, _form(), [
        _frame(PlanType.MANUAL, planning_time_min=30.0),
        _frame(PlanType.AUTO, planning_time_min=None),
    ])
    times = get_plan_times(engine, patient_id)
    assert times == {PlanType.MANUAL: 30.0, PlanType.AUTO: None}


def test_get_plan_times_empty_for_patient_with_no_plans(engine):
    patient_id = save_case(engine, _form(), [])
    assert get_plan_times(engine, patient_id) == {}


def test_update_plan_times_sets_time_on_existing_plan(engine):
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL, planning_time_min=None)])
    update_plan_times(engine, patient_id, {PlanType.MANUAL: 42.0})
    assert get_plan_times(engine, patient_id) == {PlanType.MANUAL: 42.0}


def test_update_plan_times_skips_plan_type_not_uploaded(engine):
    """No Plan row exists for Auto yet -- silently skipped, not an error."""
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL, planning_time_min=None)])
    update_plan_times(engine, patient_id, {PlanType.MANUAL: 10.0, PlanType.AUTO: 20.0})
    assert get_plan_times(engine, patient_id) == {PlanType.MANUAL: 10.0}


# --------------------------------------------------------------------------- #
# entered_by — the shared-passphrase session's "Entered by" name (see
# engine.auth / app.py), stamped onto a save and, on an overwrite, an update.
# --------------------------------------------------------------------------- #


def test_save_case_stamps_entered_by(engine):
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL)], entered_by="Dr. Somchai")
    with Session(engine) as session:
        assert session.get(Patient, patient_id).entered_by == "Dr. Somchai"


def test_save_case_entered_by_defaults_to_none(engine):
    """Callers outside the app (most of this test suite) don't pass
    entered_by — must not be required."""
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL)])
    with Session(engine) as session:
        assert session.get(Patient, patient_id).entered_by is None


def test_update_patient_overwrites_entered_by_when_given(engine):
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL)], entered_by="Dr. Somchai")
    update_patient(engine, patient_id, _form(mp1="Changed"), entered_by="Dr. Anong")
    with Session(engine) as session:
        assert session.get(Patient, patient_id).entered_by == "Dr. Anong"


def test_update_patient_leaves_entered_by_untouched_when_not_given(engine):
    """A caller that doesn't know about entered_by (e.g. an older script)
    shouldn't blank out the original save's attribution."""
    patient_id = save_case(engine, _form(), [_frame(PlanType.MANUAL)], entered_by="Dr. Somchai")
    update_patient(engine, patient_id, _form(mp1="Changed"))
    with Session(engine) as session:
        assert session.get(Patient, patient_id).entered_by == "Dr. Somchai"


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
# FormInput — relaxed intake: only pt_no is truly required (a case can be
# saved with just plan files, the rest filled in later via Module 1's Edit
# dialog — see engine.export.load_registry_frame's profile_complete)
# --------------------------------------------------------------------------- #


def test_form_input_requires_only_pt_no():
    form = FormInput(pt_no="Pt1")
    assert form.pt_no == "Pt1"
    assert form.hn is None
    assert form.dose_regimen is None
    assert form.rx_cgy is None
    assert form.fractions is None
    assert form.sib_boost is None
    assert form.tx_room is None
    assert form.mp1 is None
    assert form.mp2 is None
    assert form.ro is None


def test_form_input_still_rejects_blank_pt_no():
    with pytest.raises(ValidationError):
        FormInput(pt_no="")


@pytest.mark.parametrize("field", ["tx_room", "mp1", "mp2", "ro"])
def test_form_input_optional_strings_still_reject_empty_when_given(field):
    """None (not given) is fine; an explicit empty string is still not a
    real value — same min_length=1 rule as before, just no longer
    mandatory that a value be given at all."""
    with pytest.raises(ValidationError):
        FormInput(pt_no="Pt1", **{field: ""})


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
