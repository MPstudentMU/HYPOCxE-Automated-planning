"""End-to-end integration between engine/scoring.py (Phase 9a) and the
already-shipped engine/priority_filter.py (Module 4), exactly as
pages/4_plan_quality.py's ADAPTER section wires them together.

Covers what this task explicitly asks to confirm: the goal-scores
DataFrame carries every column engine/priority_filter.py needs, and
"All Priorities" reproduces the same Quality Index as summing
compute_goal_scores()'s own output with no filter applied at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import analysis as A
from engine import corrections as C
from engine import imputation as I
from engine import parser as P
from engine.priority_filter import PrioritySelection, REQUIRED_COLUMNS, quality_index_table
from engine.schemas import DoseRegimen, FormInput
from engine.storage import init_db, load_analysis_frame, save_case

FIXTURES = "tests/fixtures/pilot"


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


@pytest.fixture
def pt1_pt5_engine():
    engine = init_db("sqlite://")
    parsed1 = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form(pt_no="Pt1", hn="90000001"))
    save_case(engine, _form(pt_no="Pt1", hn="90000001"), parsed1.plan_frames)

    parsed5 = P.parse_case_files([f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
                                 _form(pt_no="Pt5", hn="90000005"))
    save_case(engine, _form(pt_no="Pt5", hn="90000005"), parsed5.plan_frames)
    return engine


@pytest.fixture
def scores(pt1_pt5_engine):
    pipeline_df = I.apply_straight_pass(C.apply_corrections(load_analysis_frame(pt1_pt5_engine), pt1_pt5_engine))
    return A.compute_goal_scores(pipeline_df)


# --------------------------------------------------------------------------- #
# engine.analysis re-export — pages/4_plan_quality.py's ADAPTER section
# imports these names from engine.analysis, not engine.scoring
# --------------------------------------------------------------------------- #


def test_compute_goal_scores_importable_from_engine_analysis():
    assert A.compute_goal_scores is not None


def test_compute_critical_alerts_importable_from_engine_analysis():
    assert A.compute_critical_alerts is not None


def test_reexports_are_the_same_function_objects_as_scoring():
    from engine import scoring as S
    assert A.compute_goal_scores is S.compute_goal_scores
    assert A.compute_critical_alerts is S.compute_critical_alerts


# --------------------------------------------------------------------------- #
# Column contract: engine/priority_filter.py's REQUIRED_COLUMNS, by name
# --------------------------------------------------------------------------- #


def test_goal_scores_has_every_column_priority_filter_requires(scores):
    assert REQUIRED_COLUMNS.issubset(set(scores.columns))


def test_goal_scores_has_structure_group_for_the_radar_chart(scores):
    """Not in REQUIRED_COLUMNS (radar_table degrades gracefully without
    it), but the task's own column list — and the manual's §3.7 radar
    chart — need it present."""
    assert "structure_group" in scores.columns


def test_priority_filter_functions_accept_compute_goal_scores_output_directly(scores):
    """No adapting/renaming needed between the two modules — validate_
    against REQUIRED_COLUMNS (called internally by every priority_filter
    function) doesn't raise."""
    from engine.priority_filter import category_breakdown, radar_table

    selection = PrioritySelection.all()
    quality_index_table(scores, selection)      # raises KeyError if a column's missing
    category_breakdown(scores, selection)
    radar_table(scores, selection)


# --------------------------------------------------------------------------- #
# "All Priorities" == compute_goal_scores() summed with no filter
# --------------------------------------------------------------------------- #


def test_all_priorities_equals_unfiltered_sum_of_compute_goal_scores(scores):
    qi = quality_index_table(scores, PrioritySelection.all())

    scorable = scores[scores["scorable"].astype(bool)]
    manual = (scorable.groupby(["patient", "plan"], as_index=False)
                      .agg(score=("weighted_score", "sum"), max_score=("max_weighted", "sum")))
    manual["quality_index"] = manual["score"] / manual["max_score"] * 100.0

    merged = qi.merge(manual, on=["patient", "plan"], suffixes=("_qi", "_manual"))
    assert len(merged) == len(qi) == len(manual)
    assert np.allclose(merged["quality_index_qi"], merged["quality_index_manual"])


def test_all_priorities_hand_checked_pt1_manual(scores):
    """Pt1 Manual, no filter: BODY (Hot-spot, weighted 0) + Bladder (OAR,
    weighted 0) + Kidney Lt (OAR P2, weighted 4) + PTV (Target P1,
    weighted 15) = score 19, out of max 3+9+6+15=33 -> 57.575...%.
    (Bone Marrow and Rectum are non-scorable — pending correction — and
    excluded from both numerator and denominator.)"""
    qi = quality_index_table(scores, PrioritySelection.all())
    row = qi[(qi.patient == "Pt1") & (qi.plan == "Manual")].iloc[0]
    assert row.score == pytest.approx(19.0)
    assert row.max_score == pytest.approx(33.0)
    assert row.quality_index == pytest.approx(19.0 / 33.0 * 100.0)


def test_single_priority_selection_is_a_strict_subset_of_all_priorities(scores):
    """Filtering to just Priority 1 should never sum more goals than "All
    Priorities" does for the same patient/plan."""
    qi_all = quality_index_table(scores, PrioritySelection.all())
    qi_p1 = quality_index_table(scores, PrioritySelection((1,)))

    merged = qi_all.merge(qi_p1, on=["patient", "plan"], suffixes=("_all", "_p1"))
    assert (merged["n_goals_p1"] <= merged["n_goals_all"]).all()
