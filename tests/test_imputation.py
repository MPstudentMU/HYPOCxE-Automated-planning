"""Tests for engine/imputation.py (the Straight-Pass Exception Rule),
against the real pilot fixtures — Pt5 has only an Auto file (the
straight-pass case), Pt1 has Manual+Auto (also qualifies, since neither
uploads a real Auto+Manual plan)."""
from __future__ import annotations

import pandas as pd
import pytest

from engine import corrections as C
from engine import imputation as I
from engine import parser as P
from engine.schemas import DoseRegimen, FormInput, PlanType
from engine.storage import init_db, load_analysis_frame, save_case

FIXTURES = "tests/fixtures/pilot"


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


@pytest.fixture
def engine():
    return init_db("sqlite://")


def _load_case(engine, files, form) -> None:
    parsed = P.parse_case_files(files, form)
    save_case(engine, form, parsed.plan_frames)


def _corrected_frame(engine) -> pd.DataFrame:
    return C.apply_corrections(load_analysis_frame(engine), engine)


# --------------------------------------------------------------------------- #
# needs_straight_pass
# --------------------------------------------------------------------------- #


def test_needs_straight_pass_auto_only(engine):
    _load_case(engine, [f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
              _form(pt_no="Pt5", hn="90000005"))
    df = _corrected_frame(engine)
    assert bool(I.needs_straight_pass(df[df["pt_no"] == "Pt5"])) is True


def test_needs_straight_pass_false_when_no_auto_plan(engine):
    _load_case(engine, [f"{FIXTURES}/4clinical_goals_90000004_Manual.xlsx"],
              _form(pt_no="Pt4", hn="90000004"))
    df = _corrected_frame(engine)
    assert bool(I.needs_straight_pass(df[df["pt_no"] == "Pt4"])) is False


def test_needs_straight_pass_false_when_auto_manual_has_data():
    df = pd.DataFrame([
        dict(pt_no="X", plan_type="Auto", achieved_value=1.0),
        dict(pt_no="X", plan_type="Auto+Manual", achieved_value=2.0),
    ])
    assert bool(I.needs_straight_pass(df)) is False


def test_needs_straight_pass_true_when_auto_manual_entirely_blank():
    df = pd.DataFrame([
        dict(pt_no="X", plan_type="Auto", achieved_value=1.0),
        dict(pt_no="X", plan_type="Auto+Manual", achieved_value=None),
        dict(pt_no="X", plan_type="Auto+Manual", achieved_value=None),
    ])
    assert bool(I.needs_straight_pass(df)) is True


# --------------------------------------------------------------------------- #
# apply_straight_pass — Pt5 (real fixture, straight-pass)
# --------------------------------------------------------------------------- #


def test_pt5_straight_pass_copies_auto_into_auto_manual(engine):
    _load_case(engine, [f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
              _form(pt_no="Pt5", hn="90000005"))
    df = _corrected_frame(engine)
    pt5 = df[df["pt_no"] == "Pt5"]
    assert set(pt5["plan_type"]) == {"Auto"}  # no Auto+Manual before imputation

    imputed = I.apply_straight_pass(df)
    pt5_imputed = imputed[imputed["pt_no"] == "Pt5"]

    assert set(pt5_imputed["plan_type"]) == {"Auto", "Auto+Manual"}
    auto_rows = pt5_imputed[pt5_imputed["plan_type"] == "Auto"]
    am_rows = pt5_imputed[pt5_imputed["plan_type"] == "Auto+Manual"]
    assert len(auto_rows) == len(am_rows)
    assert not auto_rows["imputed_from_auto"].any()
    assert am_rows["imputed_from_auto"].all()
    # same values, just relabeled to Auto+Manual (compare goal_key -> value
    # maps rather than sorted lists, since NaN != NaN breaks list equality)
    auto_map = dict(zip(auto_rows["goal_key"], auto_rows["achieved_value"]))
    am_map = dict(zip(am_rows["goal_key"], am_rows["achieved_value"]))
    assert auto_map.keys() == am_map.keys()
    for key in auto_map:
        a, b = auto_map[key], am_map[key]
        assert (pd.isna(a) and pd.isna(b)) or a == b


def test_pt5_straight_pass_copies_planning_time(engine):
    parsed = P.parse_case_files([f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
                                _form(pt_no="Pt5", hn="90000005", time_auto=12.5))
    save_case(engine, _form(pt_no="Pt5", hn="90000005", time_auto=12.5), parsed.plan_frames)
    df = _corrected_frame(engine)
    imputed = I.apply_straight_pass(df)
    am_rows = imputed[(imputed["pt_no"] == "Pt5") & (imputed["plan_type"] == "Auto+Manual")]
    assert (am_rows["planning_time_min"] == 12.5).all()


# --------------------------------------------------------------------------- #
# apply_straight_pass — a real (or all-blank) Auto+Manual plan is left alone
# --------------------------------------------------------------------------- #


def test_real_auto_manual_plan_is_not_overwritten(engine):
    """Pt1's fixtures include a real Auto+Manual file (1clinical_goals_
    90000001_case.xlsx) — straight-pass must not touch it."""
    _load_case(engine, [
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_case.xlsx",
    ], _form(pt_no="Pt1", hn="90000001"))
    df = _corrected_frame(engine)
    before = df[(df["pt_no"] == "Pt1") & (df["plan_type"] == "Auto+Manual")]
    assert not before.empty
    assert before["achieved_value"].notna().all()

    imputed = I.apply_straight_pass(df)
    after = imputed[(imputed["pt_no"] == "Pt1") & (imputed["plan_type"] == "Auto+Manual")]
    assert not after["imputed_from_auto"].any()
    assert sorted(after["achieved_value"]) == sorted(before["achieved_value"])


def test_pt1_manual_and_auto_only_also_gets_straight_pass(engine):
    """Pt1's Manual+Auto files alone (no Auto+Manual upload) also qualify —
    confirms the rule isn't special-cased to Pt5."""
    _load_case(engine, [
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form(pt_no="Pt1", hn="90000001"))
    df = _corrected_frame(engine)
    assert "Auto+Manual" not in set(df[df["pt_no"] == "Pt1"]["plan_type"])

    imputed = I.apply_straight_pass(df)
    pt1_am = imputed[(imputed["pt_no"] == "Pt1") & (imputed["plan_type"] == "Auto+Manual")]
    assert not pt1_am.empty
    assert pt1_am["imputed_from_auto"].all()


# --------------------------------------------------------------------------- #
# Multi-patient: imputation is per-patient, doesn't cross-contaminate
# --------------------------------------------------------------------------- #


def test_imputation_is_per_patient(engine):
    _load_case(engine, [f"{FIXTURES}/5clinical_goals_90000005_auto.xlsx"],
              _form(pt_no="Pt5", hn="90000005"))
    _load_case(engine, [
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_case.xlsx",
    ], _form(pt_no="Pt1", hn="90000001"))

    df = _corrected_frame(engine)
    imputed = I.apply_straight_pass(df)

    pt5_am = imputed[(imputed["pt_no"] == "Pt5") & (imputed["plan_type"] == "Auto+Manual")]
    pt1_am = imputed[(imputed["pt_no"] == "Pt1") & (imputed["plan_type"] == "Auto+Manual")]
    assert pt5_am["imputed_from_auto"].all()
    assert not pt1_am["imputed_from_auto"].any()  # Pt1 has a real Auto+Manual file


def test_apply_straight_pass_empty_df():
    df = pd.DataFrame(columns=["pt_no", "plan_type", "achieved_value"])
    result = I.apply_straight_pass(df)
    assert result.empty
    assert "imputed_from_auto" in result.columns
