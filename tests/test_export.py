"""Tests for engine/export.py (registry frame + HN masking) and the two
registry charts in engine/charts.py."""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from engine import charts as CH
from engine import corrections as C
from engine.export import load_registry_frame, mask_hn
from engine.schemas import (
    CorrectionField, CorrectionSource, CorrectionStatus, CriteriaDirection,
    DoseRegimen, FormInput, GoalRow, GoalStatus, PlanFrame, PlanType, StructureClass,
)
from engine.storage import GoalResult, Plan, init_db, load_analysis_frame, save_case


def _form(**overrides):
    defaults = dict(
        hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400, fractions=20,
        sib_boost=False, tx_room="Room1", mp1="Alice", mp2="Bob", ro="Dr. Carter",
    )
    defaults.update(overrides)
    return FormInput(**defaults)


def _goal(**overrides) -> GoalRow:
    defaults = dict(
        goal_key="k1", priority=1, roi_raw="Bladder", roi="Bladder",
        goal_text="D0.03cc <= 4725 cGy", goal_type="DoseAtAbsoluteVolume",
        criteria=CriteriaDirection.AT_MOST, acceptance_level=4725.0, parameter_value=0.03,
        achieved_value=4700.0, status=GoalStatus.PASS, evaluable=True,
        structure_class=StructureClass.OAR,
    )
    defaults.update(overrides)
    return GoalRow(**defaults)


# --------------------------------------------------------------------------- #
# mask_hn
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hn,expected", [
    ("54687146", "****7146"),
    ("123456789", "*****6789"),
    ("1234", "****"),
    ("12", "**"),
    ("", ""),
])
def test_mask_hn(hn, expected):
    assert mask_hn(hn) == expected


def test_mask_hn_never_leaks_more_than_last_four_digits():
    masked = mask_hn("54687146")
    assert "5468" not in masked
    assert masked.endswith("7146")


# --------------------------------------------------------------------------- #
# load_registry_frame
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine():
    return init_db("sqlite://")


def test_load_registry_frame_empty_db(engine):
    df = load_registry_frame(engine)
    assert df.empty
    assert "hn" in df.columns  # unlike load_analysis_frame, this one does carry it


def test_load_registry_frame_basic_fields(engine):
    save_case(engine, _form(), [PlanFrame(plan_type=PlanType.MANUAL, planning_time_min=30.0, goals=[_goal()])])

    df = load_registry_frame(engine)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["pt_no"] == "Pt1"
    assert row["hn"] == "90000001"
    assert row["dose_regimen"] == "Hypo"
    assert row["sib_boost"] == False  # noqa: E712
    assert row["tx_room"] == "Room1"
    assert row["mp1"] == "Alice"
    assert row["mp2"] == "Bob"
    assert row["ro"] == "Dr. Carter"
    assert row["plans_uploaded"] == "Manual"
    assert row["straight_pass"] is None  # no Auto+Manual plan uploaded
    assert row["pending_count"] == 0
    assert row["created_at"] is not None


def test_load_registry_frame_plans_uploaded_in_canonical_order(engine):
    save_case(engine, _form(), [
        PlanFrame(plan_type=PlanType.AUTO_MANUAL, goals=[_goal(goal_key="k1")]),
        PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal(goal_key="k2")]),
        PlanFrame(plan_type=PlanType.AUTO, goals=[_goal(goal_key="k3")]),
    ])
    df = load_registry_frame(engine)
    assert df.iloc[0]["plans_uploaded"] == "Manual, Auto, Auto+Manual"


def test_load_registry_frame_straight_pass_flag(engine):
    save_case(engine, _form(), [PlanFrame(plan_type=PlanType.AUTO_MANUAL, is_straight_pass=True,
                                          goals=[_goal()])])
    df = load_registry_frame(engine)
    assert bool(df.iloc[0]["straight_pass"]) is True


def test_load_registry_frame_pending_count(engine):
    save_case(engine, _form(), [PlanFrame(plan_type=PlanType.MANUAL, goals=[
        _goal(goal_key="k1", achieved_value=None, status=None, evaluable=False),
        _goal(goal_key="k2", priority=None),
        _goal(goal_key="k3"),  # fully resolved
    ])])
    df = load_registry_frame(engine)
    assert df.iloc[0]["pending_count"] == 2


def test_load_registry_frame_pending_count_drops_after_correction(engine):
    save_case(engine, _form(), [PlanFrame(plan_type=PlanType.MANUAL, goals=[
        _goal(goal_key="k1", achieved_value=None, status=None, evaluable=False),
    ])])
    goal_id = int(load_analysis_frame(engine).iloc[0]["goal_id"])
    C.record_correction(engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=4700.0)
    df = load_registry_frame(engine)
    assert df.iloc[0]["pending_count"] == 0


def test_load_registry_frame_multiple_patients(engine):
    save_case(engine, _form(pt_no="Pt1", hn="90000001"),
             [PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal()])])
    save_case(engine, _form(pt_no="Pt2", hn="90000002", mp1="Carol"),
             [PlanFrame(plan_type=PlanType.AUTO, goals=[_goal()])])
    df = load_registry_frame(engine)
    assert set(df["pt_no"]) == {"Pt1", "Pt2"}
    assert set(df["mp1"]) == {"Alice", "Carol"}


# --------------------------------------------------------------------------- #
# Charts — smoke tests: they build a Figure with the right shape of data
# --------------------------------------------------------------------------- #


@pytest.fixture
def registry_df(engine):
    save_case(engine, _form(pt_no="Pt1", hn="90000001", dose_regimen=DoseRegimen.HYPO, mp1="Alice"),
             [PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal()])])
    save_case(engine, _form(pt_no="Pt2", hn="90000002", dose_regimen=DoseRegimen.HYPO, mp1="Alice"),
             [PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal()])])
    save_case(engine, _form(pt_no="Pt3", hn="90000003", dose_regimen=DoseRegimen.CONV, mp1="Carol"),
             [PlanFrame(plan_type=PlanType.MANUAL, goals=[_goal()])])
    return load_registry_frame(engine)


def test_cases_per_regimen_chart_counts(registry_df):
    fig = CH.cases_per_regimen_chart(registry_df)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    counts = dict(zip(bar.x, bar.y))
    assert counts == {"Conv": 1, "Hypo": 2}


def test_cases_per_planner_chart_counts(registry_df):
    fig = CH.cases_per_planner_chart(registry_df)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    counts = dict(zip(bar.y, bar.x))
    assert counts == {"Alice": 2, "Carol": 1}


def test_charts_have_single_axis_no_dual_axis(registry_df):
    """One measure (case count) -> one y-axis; never a second/secondary axis."""
    for fig in (CH.cases_per_regimen_chart(registry_df), CH.cases_per_planner_chart(registry_df)):
        assert "yaxis2" not in fig.to_dict()["layout"]
