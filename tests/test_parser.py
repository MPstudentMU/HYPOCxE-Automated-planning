"""Tests for engine/parser.py, against the synthetic pilot fixtures in
tests/fixtures/pilot/ (see that directory's README for what each file
exercises and why they're synthetic rather than real exports)."""
from __future__ import annotations

import glob
import os

import pandas as pd
import pytest

from engine import parser as P
from engine.schemas import CriteriaDirection, DoseRegimen, FormInput, GoalStatus, PlanType, StructureClass
from engine.storage import init_db, load_analysis_frame, save_case

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "pilot")


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURES, name)


def _fixture_bytes(name: str) -> bytes:
    with open(_fixture_path(name), "rb") as fh:
        return fh.read()


def _form(**overrides) -> FormInput:
    defaults = dict(
        hn=None,
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


# --------------------------------------------------------------------------- #
# Every pilot file loads
# --------------------------------------------------------------------------- #


def test_every_pilot_file_loads():
    files = sorted(glob.glob(os.path.join(FIXTURES, "*.xlsx")))
    assert len(files) >= 6, "expected the synthetic pilot fixtures to be present"
    for path in files:
        name = os.path.basename(path)
        warnings: list[str] = []
        df = P.parse_workbook(name, open(path, "rb").read(), form_pt_no=None, warnings=warnings)
        assert not df.empty, f"{name} produced no rows"
        assert df["Pt"].notna().all()
        assert df["Plan"].isin([p.value for p in PlanType]).all()


# --------------------------------------------------------------------------- #
# Pt5 dedupe count
# --------------------------------------------------------------------------- #


def test_pt5_auto_dedupe_count_is_correct():
    """5clinical_goals_90000005_auto.xlsx was built with 5 unique goal rows
    plus 3 exact repeats (8 rows total) — dedupe must remove exactly the 3
    repeats, leaving 5. (Stands in for the real Pt5 pilot file's much larger
    double-export, which lives outside this repo — see the fixtures README.)
    """
    warnings: list[str] = []
    df = P.parse_workbook("5clinical_goals_90000005_auto.xlsx",
                          _fixture_bytes("5clinical_goals_90000005_auto.xlsx"),
                          form_pt_no=None, warnings=warnings)
    assert len(df) == 5
    assert any("removed 3 exact duplicate row" in w for w in warnings)


def test_pt5_auto_patient_and_plan_from_filename():
    warnings: list[str] = []
    df = P.parse_workbook("5clinical_goals_90000005_auto.xlsx",
                          _fixture_bytes("5clinical_goals_90000005_auto.xlsx"),
                          form_pt_no=None, warnings=warnings)
    assert set(df["Pt"]) == {"Pt5"}
    assert set(df["Plan"]) == {"Auto"}


# --------------------------------------------------------------------------- #
# goal_key matches across Manual/Auto for Pt1
# --------------------------------------------------------------------------- #


def test_goal_key_matches_across_manual_and_auto_for_pt1():
    manual = P.parse_workbook("1clinical_goals_90000001_Manual.xlsx",
                              _fixture_bytes("1clinical_goals_90000001_Manual.xlsx"),
                              form_pt_no=None, warnings=[])
    auto = P.parse_workbook("1clinical_goals_90000001_Auto.xlsx",
                            _fixture_bytes("1clinical_goals_90000001_Auto.xlsx"),
                            form_pt_no=None, warnings=[])

    manual_key = manual.loc[manual["ROI"] == "BODY", "goal_key"].iloc[0]
    auto_key = auto.loc[auto["ROI"] == "BODY", "goal_key"].iloc[0]

    assert manual_key == auto_key == "BODY|DoseAtVolume|AtMost|4820.0000|0.0000"
    # AchievedValue legitimately differs between plans; goal_key must not.
    assert (manual.loc[manual["ROI"] == "BODY", "AchievedValue"].iloc[0]
            != auto.loc[auto["ROI"] == "BODY", "AchievedValue"].iloc[0])


# --------------------------------------------------------------------------- #
# HN-mismatch rejection
# --------------------------------------------------------------------------- #


def test_hn_mismatch_rejects_the_file():
    form = _form(hn="11111111", pt_no="Pt4")
    with pytest.raises(P.HNMismatchError):
        P.parse_case_files([_fixture_path("4clinical_goals_90000004_Manual.xlsx")], form)


def test_hn_matching_form_succeeds():
    form = _form(hn="90000004", pt_no="Pt4")
    parsed = P.parse_case_files([_fixture_path("4clinical_goals_90000004_Manual.xlsx")], form)
    assert parsed.resolved_hn == "90000004"


def test_hn_not_entered_but_files_agree_is_auto_filled():
    form = _form(hn=None, pt_no="Pt4")
    parsed = P.parse_case_files([_fixture_path("4clinical_goals_90000004_Manual.xlsx")], form)
    assert parsed.resolved_hn == "90000004"
    assert any("HN not entered" in w for w in parsed.warnings)


def test_hn_not_entered_and_files_disagree_raises():
    with pytest.raises(P.HNMismatchError):
        P.resolve_hn(["1clinical_goals_90000001_Manual.xlsx",
                      "4clinical_goals_90000004_Manual.xlsx"], None)


def test_hn_not_entered_and_no_filename_hn_raises():
    with pytest.raises(P.HNMismatchError):
        P.resolve_hn(["clinical_goals_no_hn_here.xlsx"], None)


# --------------------------------------------------------------------------- #
# Priority sentinel exclusion
# --------------------------------------------------------------------------- #


def test_priority_sentinel_rows_kept_but_marked_no_priority():
    """The sentinel row is kept (not dropped) with Priority normalized to
    NaN/None — engine/corrections.py needs it in goal_results to be
    reviewable; scoring/pass-rate exclude it by filtering on Priority being
    set, not by the parser having thrown it away."""
    warnings: list[str] = []
    df = P.parse_workbook("1clinical_goals_90000001_Manual.xlsx",
                          _fixture_bytes("1clinical_goals_90000001_Manual.xlsx"),
                          form_pt_no=None, warnings=warnings)
    assert (df["Priority"] == P.NO_PRIORITY_SENTINEL).sum() == 0
    sentinel_rows = df[df["ROI"] == "Rectum_new"]
    assert len(sentinel_rows) == 1
    assert pd.isna(sentinel_rows["Priority"].iloc[0])
    assert any("no protocol priority" in w for w in warnings)


def test_no_priority_sentinel_value():
    assert P.NO_PRIORITY_SENTINEL == 2147483647


# --------------------------------------------------------------------------- #
# Header whitespace
# --------------------------------------------------------------------------- #


def test_header_whitespace_is_stripped():
    # 1clinical_goals_90000001_Auto.xlsx has " Priority " / " Goal" headers.
    df = P.parse_workbook("1clinical_goals_90000001_Auto.xlsx",
                          _fixture_bytes("1clinical_goals_90000001_Auto.xlsx"),
                          form_pt_no=None, warnings=[])
    assert "Priority" in df.columns
    assert not any(c.strip() != c for c in df.columns)


# --------------------------------------------------------------------------- #
# Sheet-name-truncated plan detection
# --------------------------------------------------------------------------- #


def test_plan_type_from_truncated_sheet_name():
    """1clinical_goals_90000001_case.xlsx has no Plan column and a
    deliberately generic filename; only its sheet name, truncated to
    Excel's 31-char limit ('..._Auto+M'), carries the plan type."""
    df = P.parse_workbook("1clinical_goals_90000001_case.xlsx",
                          _fixture_bytes("1clinical_goals_90000001_case.xlsx"),
                          form_pt_no=None, warnings=[])
    assert set(df["Plan"]) == {"Auto+Manual"}


# --------------------------------------------------------------------------- #
# plan_type_override — dedicated per-slot uploads (New Case page)
# --------------------------------------------------------------------------- #


def test_plan_type_override_wins_over_filename_and_sheet_name():
    """The Auto file's own name/sheet clearly say Auto — force it into the
    Manual slot anyway and confirm the override, not the file, wins."""
    df = P.parse_workbook("1clinical_goals_90000001_Auto.xlsx",
                          _fixture_bytes("1clinical_goals_90000001_Auto.xlsx"),
                          form_pt_no=None, warnings=[], plan_type_override=PlanType.MANUAL)
    assert set(df["Plan"]) == {"Manual"}


def test_plan_type_override_ignored_for_format_b():
    """A combined (Format B) file declares several plans itself — an
    override for a single slot must not collapse them into one."""
    warnings: list[str] = []
    df = P.parse_workbook("6clinical_goals_90000006_combined.xlsx",
                          _fixture_bytes("6clinical_goals_90000006_combined.xlsx"),
                          form_pt_no=None, warnings=warnings, plan_type_override=PlanType.MANUAL)
    assert set(df["Plan"]) == {"Manual", "Auto", "Auto+Manual"}
    assert any("combined (Format B) file covering more than one plan" in w for w in warnings)


def test_parse_case_files_plan_type_overrides_per_file():
    form = FormInput(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                     fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    files = [_fixture_path("1clinical_goals_90000001_Auto.xlsx"),
            _fixture_path("1clinical_goals_90000001_Manual.xlsx")]
    # deliberately swapped: tell the parser the Auto file is Manual and vice versa
    parsed = P.parse_case_files(files, form,
                                plan_type_overrides=[PlanType.MANUAL, PlanType.AUTO])
    by_type = {pf.plan_type: pf for pf in parsed.plan_frames}
    assert by_type[PlanType.MANUAL].source_filename == "1clinical_goals_90000001_Auto.xlsx"
    assert by_type[PlanType.AUTO].source_filename == "1clinical_goals_90000001_Manual.xlsx"


def test_parse_case_files_rejects_mismatched_override_length():
    form = FormInput(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                     fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    with pytest.raises(P.ParserError):
        P.parse_case_files([_fixture_path("1clinical_goals_90000001_Manual.xlsx")], form,
                           plan_type_overrides=[PlanType.MANUAL, PlanType.AUTO])


def test_wrong_file_type_raises_unreadable_file_error():
    """A non-Excel upload (e.g. a renamed .csv/.txt) must raise the more
    specific UnreadableFileError, not just the generic ParserError, so the
    page layer can give a clearer message than pandas's own exception text."""
    form = FormInput(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                     fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    files = [("not_really.xlsx", b"this is not an excel file, just plain text")]
    with pytest.raises(P.UnreadableFileError):
        P.parse_case_files(files, form, plan_type_overrides=[PlanType.MANUAL])


def test_unreadable_file_error_is_a_parser_error():
    """Subclassing keeps any existing `except P.ParserError` catch-all working."""
    assert issubclass(P.UnreadableFileError, P.ParserError)


# --------------------------------------------------------------------------- #
# Pt-column disagreement in a single-plan file
# --------------------------------------------------------------------------- #


def test_single_plan_pt_column_disagreement_falls_back_to_filename():
    warnings: list[str] = []
    df = P.parse_workbook("3clinical_goals_90000003_Manual.xlsx",
                          _fixture_bytes("3clinical_goals_90000003_Manual.xlsx"),
                          form_pt_no=None, warnings=warnings)
    assert set(df["Pt"]) == {"Pt3"}
    assert any("different values in a single-plan file" in w for w in warnings)


def test_form_pt_no_takes_priority_over_pt_column():
    df = P.parse_workbook("3clinical_goals_90000003_Manual.xlsx",
                          _fixture_bytes("3clinical_goals_90000003_Manual.xlsx"),
                          form_pt_no="Pt99", warnings=[])
    assert set(df["Pt"]) == {"Pt99"}


# --------------------------------------------------------------------------- #
# AchievedValue NaN -> evaluable=False, not dropped
# --------------------------------------------------------------------------- #


def test_missing_achieved_value_is_kept_and_flagged_not_dropped():
    warnings: list[str] = []
    df = P.parse_workbook("1clinical_goals_90000001_Manual.xlsx",
                          _fixture_bytes("1clinical_goals_90000001_Manual.xlsx"),
                          form_pt_no=None, warnings=warnings)
    zbone_rows = df[df["roi"] == "Bone Marrow"]
    assert len(zbone_rows) == 1
    assert zbone_rows["evaluable"].iloc[0] == False  # noqa: E712
    assert pd.isna(zbone_rows["AchievedValue"].iloc[0])
    assert any("flagged for engine/corrections.py" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Format B combined workbook
# --------------------------------------------------------------------------- #


def test_format_b_combined_workbook_melts_all_plans():
    warnings: list[str] = []
    df = P.parse_workbook("6clinical_goals_90000006_combined.xlsx",
                          _fixture_bytes("6clinical_goals_90000006_combined.xlsx"),
                          form_pt_no=None, warnings=warnings)
    assert set(df["Plan"]) == {"Manual", "Auto", "Auto+Manual"}
    assert len(df) == 6  # 2 goals x 3 plans
    assert any("combined workbook (Format B) detected" in w for w in warnings)


def test_format_b_computes_status_for_plan_with_no_status_column():
    """Achieved_Auto+Manual has no matching Status_Auto+Manual column in the
    fixture — Status for that plan must be computed from AchievedValue vs
    AcceptanceLevel, not left null or dropped."""
    warnings: list[str] = []
    df = P.parse_workbook("6clinical_goals_90000006_combined.xlsx",
                          _fixture_bytes("6clinical_goals_90000006_combined.xlsx"),
                          form_pt_no=None, warnings=warnings)
    am = df[df["Plan"] == "Auto+Manual"]
    assert am["Status"].notna().all()

    body_am = am[am["ROI"] == "BODY"].iloc[0]
    assert body_am["AchievedValue"] == 4770.0
    assert body_am["AcceptanceLevel"] == 4800.0
    assert body_am["Status"] == "PASS"  # AtMost: 4770 <= 4800

    bladder_am = am[am["ROI"] == "Bladder"].iloc[0]
    assert bladder_am["AchievedValue"] == 4712.0
    assert bladder_am["AcceptanceLevel"] == 4700.0
    assert bladder_am["Status"] == "FAIL"  # AtMost: 4712 > 4700
    assert any("had no Status column value" in w for w in warnings)


# --------------------------------------------------------------------------- #
# Unit tests: small pure helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("raw,expected", [
    ("Bone", "Bone Marrow"),
    ("zBone", "Bone Marrow"),
    ("Bone Marrow", "Bone Marrow"),
    ("Kidney_L", "Kidney Lt"),
    ("Kidney Lt", "Kidney Lt"),
    ("Kidney_R", "Kidney Rt"),
    ("Femur Head Lt", "Femoral Head Lt"),
    ("Rectum_new", "Rectum"),
    ("SmallBowel", "Small Bowel"),
    # Targets collapse to their base type per docs/analysis_manual_th_v2.md
    # §2.4's harmonisation table — a patient is only ever on one dose
    # regimen, so a dose suffix is regimen noise, never a second target.
    ("1 ITV45", "ITV"),
    ("ITV45", "ITV"),
    ("zITV", "ITV"),
    ("ITV-T LR", "ITV"),
    ("PTV45", "PTV"),
    ("PTV 45", "PTV"),
    ("PTV44", "PTV"),
    ("PTV 44", "PTV"),
    ("1 PTV45", "PTV"),
    ("CTV P", "CTV"),
    # ...except a nodal boost marker, which keeps its own bucket.
    ("PTV-N55", "PTV-N"),
    ("1 PTV-N55", "PTV-N"),
    ("GTV-N 55", "GTV-N"),
])
def test_canonical_roi(raw, expected):
    assert P.canonical_roi(raw) == expected


def test_canonical_roi_collapses_same_patient_dose_variants():
    """PTV44 and PTV45 never coexist for one patient (each is on exactly
    one dose regimen) — the suffix is noise the manual says to drop, not a
    second target to keep distinct."""
    assert P.canonical_roi("PTV44") == P.canonical_roi("PTV45") == "PTV"


def test_canonical_roi_keeps_nodal_boost_distinct_from_primary_target():
    assert P.canonical_roi("PTV-N55") != P.canonical_roi("PTV45")


@pytest.mark.parametrize("roi,expected", [
    ("PTV45", StructureClass.TARGET),
    ("ptv44", StructureClass.TARGET),
    ("ITV-T LR", StructureClass.TARGET),
    ("CTV P", StructureClass.TARGET),
    ("GTV-N 55", StructureClass.TARGET),
    ("PTV-N55", StructureClass.TARGET),  # still TARGET; alert-exclusion is Module 4's job
    ("Bladder", StructureClass.OAR),
    ("BODY", StructureClass.OAR),
    ("Bone Marrow", StructureClass.OAR),
])
def test_classify_structure(roi, expected):
    assert P.classify_structure(roi) == expected


def test_make_goal_key_rounds_and_joins():
    key = P.make_goal_key("Bladder", "DoseAtVolume", CriteriaDirection.AT_MOST, 4725.00004, 0.03001)
    assert key == "Bladder|DoseAtVolume|AtMost|4725.0000|0.0300"


def test_make_goal_key_stable_across_float_noise():
    k1 = P.make_goal_key("Bladder", "DoseAtVolume", CriteriaDirection.AT_MOST, 4725.0, 0.03)
    k2 = P.make_goal_key("Bladder", "DoseAtVolume", CriteriaDirection.AT_MOST, 4725.00001, 0.0300001)
    assert k1 == k2


@pytest.mark.parametrize("achieved,acceptance,criteria,expected", [
    (4800.0, 4820.0, CriteriaDirection.AT_MOST, GoalStatus.PASS),
    (4830.0, 4820.0, CriteriaDirection.AT_MOST, GoalStatus.FAIL),
    (97.0, 95.0, CriteriaDirection.AT_LEAST, GoalStatus.PASS),
    (93.0, 95.0, CriteriaDirection.AT_LEAST, GoalStatus.FAIL),
    (None, 95.0, CriteriaDirection.AT_LEAST, None),
])
def test_compute_status(achieved, acceptance, criteria, expected):
    assert P.compute_status(achieved, acceptance, criteria) == expected


@pytest.mark.parametrize("text,expected", [
    ("Manual", PlanType.MANUAL),
    ("manual", PlanType.MANUAL),
    ("Auto", PlanType.AUTO),
    ("AUTO", PlanType.AUTO),
    ("Auto+Manual", PlanType.AUTO_MANUAL),
    ("AutoManual", PlanType.AUTO_MANUAL),
    ("Auto+M", PlanType.AUTO_MANUAL),   # Excel's 31-char sheet-name truncation
    ("auto_manual", PlanType.AUTO_MANUAL),
    ("something_else", None),
    (None, None),
])
def test_normalize_plan_token(text, expected):
    assert P.normalize_plan_token(text) == expected


def test_hn_from_filename():
    assert P.hn_from_filename("3clinical_goals_54687146_Auto.xlsx") == "54687146"
    assert P.hn_from_filename("no_hn_here.xlsx") is None
    assert P.hn_from_filename("1clinical_goals_90000001_Manual.xlsx") == "90000001"


def test_patient_no_from_text():
    assert P.patient_no_from_text("5clinical_goals_39097083_auto.xlsx") == "Pt5"
    assert P.patient_no_from_text("Pt3") == "Pt3"
    assert P.patient_no_from_text("no digits") is None


# --------------------------------------------------------------------------- #
# End-to-end: parse -> save_case -> load_analysis_frame
# --------------------------------------------------------------------------- #


def test_parsed_case_round_trips_through_storage_without_hn():
    form = _form(hn="90000001", pt_no="Pt1")
    files = [
        _fixture_path("1clinical_goals_90000001_Manual.xlsx"),
        _fixture_path("1clinical_goals_90000001_Auto.xlsx"),
        _fixture_path("1clinical_goals_90000001_case.xlsx"),
    ]
    parsed = P.parse_case_files(files, form)

    engine = init_db("sqlite://")
    save_case(engine, form, parsed.plan_frames)

    df = load_analysis_frame(engine)
    assert "hn" not in df.columns
    assert set(df["plan_type"]) == {"Manual", "Auto", "Auto+Manual"}
    assert len(df) == sum(len(pf.goals) for pf in parsed.plan_frames)
