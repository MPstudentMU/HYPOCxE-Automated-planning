"""Tests for engine/export.py (registry frame + HN masking, and the
Module 9 multi-sheet workbook export) and the two registry charts in
engine/charts.py."""
from __future__ import annotations

import io

import pandas as pd
import plotly.graph_objects as go
import pytest
from sqlmodel import Session, select

from engine import charts as CH
from engine import corrections as C
from engine import parser as P
from engine.criteria_v0 import CRITERIA_SHA256, CRITERIA_VERSION, ENGINE_VERSION
from engine.export import build_analysis_workbook, load_registry_frame, mask_hn
from engine.priority_filter import PrioritySelection
from engine.schemas import (
    CorrectionField, CorrectionSource, CorrectionStatus, CriteriaDirection,
    DoseRegimen, FormInput, GoalRow, GoalStatus, PlanFrame, PlanType, StructureClass,
)
from engine.storage import AnalysisRun, GoalResult, Plan, init_db, load_analysis_frame, save_case

FIXTURES = "tests/fixtures/pilot"


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


# --------------------------------------------------------------------------- #
# build_analysis_workbook — Module 9's one-workbook-per-run export
# --------------------------------------------------------------------------- #


def _export_form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


@pytest.fixture
def pt1_pt5_engine():
    """Real pilot fixtures: Pt1 has Manual+Auto (so pass-rate/scoring/DVH
    all have something to compare), Pt5 has an Auto-only upload — enough
    for every sheet to be non-trivially populated without being a slow,
    full-cohort fixture set."""
    engine = init_db("sqlite://")
    parsed1 = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _export_form(pt_no="Pt1", hn="90000001"))
    save_case(engine, _export_form(pt_no="Pt1", hn="90000001"), parsed1.plan_frames)

    parsed5 = P.parse_case_files([f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
                                 _export_form(pt_no="Pt5", hn="90000005"))
    save_case(engine, _export_form(pt_no="Pt5", hn="90000005"), parsed5.plan_frames)
    return engine


EXPECTED_SHEETS = [
    "RunInfo", "M1_Patients", "M2_PassRate", "M2_Detail", "M3_Time",
    "M4_QualitySummary", "M4_GoalScores", "M4_Consistency",
    "M5_DVHConsistency", "Alerts", "DataCorrections",
]


def test_build_analysis_workbook_returns_bytes(pt1_pt5_engine):
    workbook = build_analysis_workbook(pt1_pt5_engine)
    assert isinstance(workbook, bytes)
    assert len(workbook) > 0


def test_build_analysis_workbook_has_every_sheet_in_order(pt1_pt5_engine):
    workbook = build_analysis_workbook(pt1_pt5_engine)
    xls = pd.ExcelFile(io.BytesIO(workbook), engine="openpyxl")
    assert xls.sheet_names == EXPECTED_SHEETS


def test_build_analysis_workbook_masks_hn_everywhere(pt1_pt5_engine):
    """CLAUDE.md rule 4: HN never leaves the app unmasked. M1_Patients
    shows only the masked form, and the raw HN string appears nowhere in
    any sheet of the exported file."""
    workbook = build_analysis_workbook(pt1_pt5_engine)
    xls = pd.ExcelFile(io.BytesIO(workbook), engine="openpyxl")

    m1 = xls.parse("M1_Patients")
    assert set(m1["hn"]) == {"****0001", "****0005"}

    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        as_text = df.astype(str)
        assert not as_text.isin(["90000001", "90000005"]).any().any(), \
            f"raw HN leaked into sheet {sheet!r}"


def test_build_analysis_workbook_writes_one_analysis_run_row(pt1_pt5_engine):
    with Session(pt1_pt5_engine) as session:
        before = len(session.exec(select(AnalysisRun)).all())

    build_analysis_workbook(pt1_pt5_engine)

    with Session(pt1_pt5_engine) as session:
        runs = session.exec(select(AnalysisRun)).all()
    assert len(runs) == before + 1
    run = runs[-1]
    assert run.engine_version == ENGINE_VERSION
    assert run.criteria_version == CRITERIA_VERSION.upper()


def test_build_analysis_workbook_run_info_has_version_and_hash(pt1_pt5_engine):
    workbook = build_analysis_workbook(pt1_pt5_engine)
    xls = pd.ExcelFile(io.BytesIO(workbook), engine="openpyxl")
    run_info = xls.parse("RunInfo").set_index("field")["value"]

    assert run_info["Engine version"] == ENGINE_VERSION
    assert run_info["Criteria version"] == "V0"  # displayed uppercase, per the request
    assert run_info["Criteria SHA-256"] == CRITERIA_SHA256
    assert run_info["Active priority filter (M4)"] == "All Priorities"
    assert bool(run_info["Pass rate: exclude non-evaluable goals"]) is True
    assert "Export timestamp (UTC)" in run_info.index


def test_build_analysis_workbook_criteria_version_constant_is_lowercase():
    """The stored constant must stay lowercase 'v0' — the pinned
    CRITERIA_SHA256 is computed over it (engine/criteria_v0.py, CLAUDE.md
    rule 2). Only the RunInfo *display* is uppercased."""
    assert CRITERIA_VERSION == "v0"


def test_build_analysis_workbook_priority_selection_changes_m4_quality_summary(pt1_pt5_engine):
    all_workbook = build_analysis_workbook(pt1_pt5_engine, priority_selection=PrioritySelection.all())
    p1_workbook = build_analysis_workbook(pt1_pt5_engine, priority_selection=PrioritySelection((1,)))

    all_qi = pd.ExcelFile(io.BytesIO(all_workbook), engine="openpyxl").parse("M4_QualitySummary")
    p1_qi = pd.ExcelFile(io.BytesIO(p1_workbook), engine="openpyxl").parse("M4_QualitySummary")

    all_run_info = pd.ExcelFile(io.BytesIO(all_workbook), engine="openpyxl") \
        .parse("RunInfo").set_index("field")["value"]
    p1_run_info = pd.ExcelFile(io.BytesIO(p1_workbook), engine="openpyxl") \
        .parse("RunInfo").set_index("field")["value"]
    assert all_run_info["Active priority filter (M4)"] == "All Priorities"
    assert p1_run_info["Active priority filter (M4)"] == "Priority 1"

    assert not all_qi["n_goals"].equals(p1_qi["n_goals"])


def test_build_analysis_workbook_m4_goal_scores_is_unfiltered_by_priority(pt1_pt5_engine):
    """M4_GoalScores is the raw compute_goal_scores() output — every goal,
    regardless of the M4_QualitySummary priority filter."""
    all_workbook = build_analysis_workbook(pt1_pt5_engine, priority_selection=PrioritySelection.all())
    p1_workbook = build_analysis_workbook(pt1_pt5_engine, priority_selection=PrioritySelection((1,)))

    all_scores = pd.ExcelFile(io.BytesIO(all_workbook), engine="openpyxl").parse("M4_GoalScores")
    p1_scores = pd.ExcelFile(io.BytesIO(p1_workbook), engine="openpyxl").parse("M4_GoalScores")
    assert len(all_scores) == len(p1_scores)


def test_build_analysis_workbook_data_corrections_sheet_reflects_corrections(pt1_pt5_engine):
    df = load_analysis_frame(pt1_pt5_engine)
    bone_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    C.record_correction(pt1_pt5_engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=800.0)

    workbook = build_analysis_workbook(pt1_pt5_engine)
    corrections = pd.ExcelFile(io.BytesIO(workbook), engine="openpyxl").parse("DataCorrections")
    assert len(corrections) == 1
    assert corrections.iloc[0]["corrected_by"] == "Dr. Test"


def test_build_analysis_workbook_pass_rate_settings_override(pt1_pt5_engine):
    default_workbook = build_analysis_workbook(pt1_pt5_engine)
    matched_workbook = build_analysis_workbook(
        pt1_pt5_engine, pass_rate_settings=dict(matched_goals_only=True))

    default_info = pd.ExcelFile(io.BytesIO(default_workbook), engine="openpyxl") \
        .parse("RunInfo").set_index("field")["value"]
    matched_info = pd.ExcelFile(io.BytesIO(matched_workbook), engine="openpyxl") \
        .parse("RunInfo").set_index("field")["value"]
    assert bool(default_info["Pass rate: matched goals only"]) is False
    assert bool(matched_info["Pass rate: matched goals only"]) is True
