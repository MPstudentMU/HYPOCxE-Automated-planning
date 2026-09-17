"""AppTest-based tests for pages/0b_batch_upload.py — the actual page
around engine.batch's grouping/classification (tested directly, against
the same fixtures, in tests/test_batch.py). These exercise what only a
running page can show: the summary table, the Duplicate Skip/Overwrite
radio, and that clicking "Confirm batch save" commits exactly the
unblocked rows.
"""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

import components.db
from engine.export import load_registry_frame
from engine.schemas import DoseRegimen, FormInput
from engine.storage import init_db, save_case

FIXTURES = "tests/fixtures/pilot"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "pages", "0b_batch_upload.py")


def _read(name: str) -> bytes:
    with open(f"{FIXTURES}/{name}", "rb") as f:
        return f.read()


@pytest.fixture
def engine():
    return init_db("sqlite://")


def _app(engine, monkeypatch) -> AppTest:
    """A fresh AppTest for pages/0b_batch_upload.py wired to `engine`.

    Monkeypatches components.db.get_engine (imported by the page at module
    load time) rather than engine.config.DEFAULT_DB_URL: get_engine() is
    @st.cache_resource-wrapped with no arguments, so patching the URL alone
    wouldn't stop a real run from reusing an already-cached engine — this
    keeps every test off the real data/hypocxe.db.
    """
    monkeypatch.setattr(components.db, "get_engine", lambda: engine)
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    return at


def _upload(at, files: list[tuple[str, bytes]]) -> None:
    at.file_uploader[0].set_value([(name, content, XLSX_MIME) for name, content in files])
    at.run(timeout=30)


# --------------------------------------------------------------------------- #
# Three patients, mixed plan types, uploaded out of order -> correct grouping
# --------------------------------------------------------------------------- #


def test_three_patients_mixed_and_out_of_order_group_correctly(engine, monkeypatch):
    at = _app(engine, monkeypatch)
    _upload(at, [
        ("5clinical_goals_90000005_auto.xlsx", _read("5clinical_goals_90000005_auto.xlsx")),
        ("1clinical_goals_90000001_case.xlsx", _read("1clinical_goals_90000001_case.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
    ])

    assert list(at.exception) == []
    summary = at.dataframe[0].value.set_index("Patient")
    assert set(summary.index) == {"Pt1", "Pt3", "Pt5"}
    assert (summary["Status"] == "🆕 New").all()
    assert summary.loc["Pt1", "Detected plans"] == "Manual, Auto, Auto+Manual"
    assert summary.loc["Pt3", "Detected plans"] == "Manual"
    assert summary.loc["Pt5", "Detected plans"] == "Auto"
    assert summary.loc["Pt1", "HN"] == "90000001"


def test_three_patients_confirm_batch_save_creates_all_three(engine, monkeypatch):
    at = _app(engine, monkeypatch)
    _upload(at, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
        ("5clinical_goals_90000005_auto.xlsx", _read("5clinical_goals_90000005_auto.xlsx")),
    ])

    save_btn = next(b for b in at.button if b.label == "Confirm batch save")
    save_btn.click()
    at.run(timeout=30)

    assert list(at.exception) == []
    reg = load_registry_frame(engine)
    assert set(reg["pt_no"]) == {"Pt1", "Pt3", "Pt5"}


# --------------------------------------------------------------------------- #
# An existing pt_no is flagged Duplicate, not silently overwritten
# --------------------------------------------------------------------------- #


def test_existing_patient_flagged_duplicate_defaults_to_skip(engine, monkeypatch):
    save_case(engine, FormInput(pt_no="Pt1", hn="90000001", dose_regimen=DoseRegimen.HYPO,
                                rx_cgy=4400, fractions=20, sib_boost=False), [])

    at = _app(engine, monkeypatch)
    _upload(at, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])

    summary = at.dataframe[0].value.set_index("Patient")
    assert "Duplicate" in summary.loc["Pt1", "Status"]
    radio = next(r for r in at.radio if "already exists" in r.label)
    assert radio.value == "Skip"  # safe default -- not silently overwritten


def test_duplicate_left_as_skip_confirm_batch_save_does_not_touch_it(engine, monkeypatch):
    """With every group Skipped (the default), there's nothing actionable
    to save -- Confirm batch save is correctly disabled, exactly like a
    browser user would see it (AppTest itself refuses to click a disabled
    button), so the existing patient's plans are left untouched."""
    from sqlmodel import Session, select
    from engine.storage import Plan

    save_case(engine, FormInput(pt_no="Pt1", hn="90000001", dose_regimen=DoseRegimen.HYPO,
                                rx_cgy=4400, fractions=20, sib_boost=False), [])

    at = _app(engine, monkeypatch)
    _upload(at, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    save_btn = next(b for b in at.button if b.label == "Confirm batch save")
    assert save_btn.disabled is True

    with Session(engine) as session:
        assert session.exec(select(Plan)).all() == []  # left untouched, per Skip


def test_duplicate_set_to_overwrite_confirm_batch_save_replaces_its_plans(engine, monkeypatch):
    save_case(engine, FormInput(pt_no="Pt1", hn="90000001", dose_regimen=DoseRegimen.HYPO,
                                rx_cgy=4400, fractions=20, sib_boost=False), [])

    at = _app(engine, monkeypatch)
    _upload(at, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    radio = next(r for r in at.radio if "already exists" in r.label)
    radio.set_value("Overwrite")
    at.run(timeout=30)

    save_btn = next(b for b in at.button if b.label == "Confirm batch save")
    save_btn.click()
    at.run(timeout=30)

    assert list(at.exception) == []
    from engine.storage import load_analysis_frame
    df = load_analysis_frame(engine)
    assert set(df.loc[df["pt_no"] == "Pt1", "plan_type"]) == {"Manual", "Auto"}


# --------------------------------------------------------------------------- #
# One patient's files carry two different HNs -> Needs review, blocked
# alone; the rest of the batch still saves via Confirm batch save
# --------------------------------------------------------------------------- #


def test_multiple_hns_needs_review_others_still_save(engine, monkeypatch):
    manual_content = _read("1clinical_goals_90000001_Manual.xlsx")
    auto_content = _read("1clinical_goals_90000001_Auto.xlsx")

    at = _app(engine, monkeypatch)
    _upload(at, [
        ("3clinical_goals_11111111_Manual.xlsx", manual_content),
        ("3clinical_goals_22222222_Auto.xlsx", auto_content),
        ("5clinical_goals_90000005_auto.xlsx", _read("5clinical_goals_90000005_auto.xlsx")),
    ])

    summary = at.dataframe[0].value.set_index("Patient")
    assert "Needs review" in summary.loc["Pt3", "Status"]
    assert summary.loc["Pt5", "Status"] == "🆕 New"

    save_btn = next(b for b in at.button if b.label == "Confirm batch save")
    save_btn.click()
    at.run(timeout=30)

    assert list(at.exception) == []
    reg = load_registry_frame(engine)
    assert set(reg["pt_no"]) == {"Pt5"}  # Pt3 blocked, never saved
