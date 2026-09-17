"""Tests for engine/batch.py: detect_batch_files() and group_batch_files()
— the non-Streamlit grouping/classification logic behind
pages/0b_batch_upload.py, against real tests/fixtures/pilot/ files."""
from __future__ import annotations

import os

import pytest

from engine.batch import BatchFile, detect_batch_files, group_batch_files
from engine.schemas import DoseRegimen, FormInput, PlanType
from engine.storage import init_db, save_case

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "pilot")


def _read(name: str) -> bytes:
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


@pytest.fixture
def engine():
    return init_db("sqlite://")


# --------------------------------------------------------------------------- #
# detect_batch_files — reuses engine.parser's own detection, doesn't
# duplicate it (see tests/test_parser.py for that cascade's own coverage)
# --------------------------------------------------------------------------- #


def test_detect_batch_files_extracts_pt_no_hn_and_plan_type():
    files = [("1clinical_goals_90000001_Manual.xlsx",
              _read("1clinical_goals_90000001_Manual.xlsx"))]
    detected = detect_batch_files(files)
    assert len(detected) == 1
    f = detected[0]
    assert f.pt_no == "Pt1"
    assert f.hn == "90000001"
    assert f.plan_type == PlanType.MANUAL


def test_detect_batch_files_leaves_undetectable_fields_none():
    """6clinical_goals_90000006_combined.xlsx: pt_no/hn detect fine from
    the filename, but plan_type doesn't (Format B, no single plan)."""
    files = [("6clinical_goals_90000006_combined.xlsx",
              _read("6clinical_goals_90000006_combined.xlsx"))]
    detected = detect_batch_files(files)
    assert detected[0].pt_no == "Pt6"
    assert detected[0].hn == "90000006"
    assert detected[0].plan_type is None


# --------------------------------------------------------------------------- #
# group_batch_files — 3 patients' files together, mixed types, out of order
# --------------------------------------------------------------------------- #


def test_group_batch_files_three_patients_mixed_and_out_of_order(engine):
    """Pt1 (Manual+Auto+Auto+Manual), Pt3 (Manual only), Pt5 (Auto only),
    deliberately not uploaded in patient order or plan-type order."""
    files = [
        ("5clinical_goals_90000005_auto.xlsx", _read("5clinical_goals_90000005_auto.xlsx")),
        ("1clinical_goals_90000001_case.xlsx", _read("1clinical_goals_90000001_case.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
    ]
    detected = detect_batch_files(files)
    assert all(f.pt_no is not None and f.plan_type is not None for f in detected)

    groups = group_batch_files(engine, detected)
    by_pt_no = {g.pt_no: g for g in groups}

    assert set(by_pt_no) == {"Pt1", "Pt3", "Pt5"}
    assert all(g.status == "new" for g in groups)

    assert len(by_pt_no["Pt1"].files) == 3
    assert {f.plan_type for f in by_pt_no["Pt1"].files} == {
        PlanType.MANUAL, PlanType.AUTO, PlanType.AUTO_MANUAL,
    }
    assert by_pt_no["Pt1"].resolved_hn == "90000001"

    assert len(by_pt_no["Pt3"].files) == 1
    assert by_pt_no["Pt3"].resolved_hn == "90000003"

    assert len(by_pt_no["Pt5"].files) == 1
    assert by_pt_no["Pt5"].resolved_hn == "90000005"


# --------------------------------------------------------------------------- #
# group_batch_files — an existing pt_no is flagged Duplicate, not silently
# overwritten
# --------------------------------------------------------------------------- #


def test_group_batch_files_flags_existing_pt_no_as_duplicate(engine):
    existing_id = save_case(engine, _form(pt_no="Pt1", hn="90000001"), [])

    files = [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ]
    detected = detect_batch_files(files)
    groups = group_batch_files(engine, detected)

    assert len(groups) == 1
    group = groups[0]
    assert group.pt_no == "Pt1"
    assert group.status == "duplicate"
    assert group.existing_patient_id == existing_id
    assert group.resolved_hn == "90000001"


def test_group_batch_files_duplicate_does_not_touch_the_database(engine):
    """group_batch_files only classifies — it must never call save_case,
    update_patient, or replace_plan itself; that's pages/0b_batch_upload.py's
    job, gated on the user's Skip/Overwrite choice."""
    from sqlmodel import Session, select
    from engine.storage import GoalResult, Plan

    save_case(engine, _form(pt_no="Pt1", hn="90000001"), [])
    files = [("1clinical_goals_90000001_Manual.xlsx",
              _read("1clinical_goals_90000001_Manual.xlsx"))]
    group_batch_files(engine, detect_batch_files(files))

    with Session(engine) as session:
        assert session.exec(select(Plan)).all() == []
        assert session.exec(select(GoalResult)).all() == []


# --------------------------------------------------------------------------- #
# group_batch_files — one patient's files carry two different HNs -> Needs
# review, blocked alone, the rest of the batch still saves
# --------------------------------------------------------------------------- #


def test_group_batch_files_multiple_hns_needs_review_others_unaffected(engine):
    """Pt3's two files disagree on HN (11111111 vs 22222222) — real,
    already-parseable fixture content (Pt1's Manual/Auto exports), just
    wrapped in filenames that collide on patient number but not HN.
    Pt5's own, entirely separate file must still classify "new"."""
    manual_content = _read("1clinical_goals_90000001_Manual.xlsx")
    auto_content = _read("1clinical_goals_90000001_Auto.xlsx")

    files = [
        ("3clinical_goals_11111111_Manual.xlsx", manual_content),
        ("3clinical_goals_22222222_Auto.xlsx", auto_content),
        ("5clinical_goals_90000005_auto.xlsx", _read("5clinical_goals_90000005_auto.xlsx")),
    ]
    detected = detect_batch_files(files)
    # sanity: detection itself extracted two distinct HNs for Pt3's files
    pt3_hns = {f.hn for f in detected if f.pt_no == "Pt3"}
    assert pt3_hns == {"11111111", "22222222"}

    groups = group_batch_files(engine, detected)
    by_pt_no = {g.pt_no: g for g in groups}

    assert by_pt_no["Pt3"].status == "needs_review"
    assert "HN" in by_pt_no["Pt3"].reason or "hn" in by_pt_no["Pt3"].reason.lower()
    assert by_pt_no["Pt3"].resolved_hn is None

    assert by_pt_no["Pt5"].status == "new"
    assert by_pt_no["Pt5"].resolved_hn == "90000005"


def test_group_batch_files_needs_review_for_duplicate_plan_type_within_group(engine):
    """Defensive fallback: if a group somehow still has two files sharing
    a plan_type (the page is meant to resolve this before calling here),
    group_batch_files must flag it rather than pick one arbitrarily."""
    files = [
        BatchFile(name="a.xlsx", content=b"", pt_no="Pt9", hn="90000009", plan_type=PlanType.MANUAL),
        BatchFile(name="b.xlsx", content=b"", pt_no="Pt9", hn="90000009", plan_type=PlanType.MANUAL),
    ]
    groups = group_batch_files(engine, files)
    assert len(groups) == 1
    assert groups[0].status == "needs_review"
    assert "plan type" in groups[0].reason.lower()
