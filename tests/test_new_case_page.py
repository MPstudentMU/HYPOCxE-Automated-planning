"""AppTest-based tests for pages/0_new_case.py's multi-file upload and
plan-type detection/conflict-resolution UI.

engine.parser.detect_plan_type() itself (the Plan column -> sheet name ->
filename cascade) is tested directly in tests/test_parser.py against real
values. These tests exercise the page around it — the file-uploader,
selectbox and Preview-button interplay — which can't be verified any other
way without a live browser (this project's usual fallback for page-level
checks, but Streamlit's AppTest harness can drive this exact interaction
model — upload, read widget state, click, rerun — without one).
"""
from __future__ import annotations

import os

import pytest
from streamlit.testing.v1 import AppTest

import components.db
from engine.schemas import PlanType
from engine.storage import init_db

FIXTURES = "tests/fixtures/pilot"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "pages", "0_new_case.py")


def _read(name: str) -> bytes:
    with open(f"{FIXTURES}/{name}", "rb") as f:
        return f.read()


@pytest.fixture
def app(monkeypatch):
    """A fresh in-memory-db AppTest instance for pages/0_new_case.py.

    Monkeypatches components.db.get_engine (imported by the page at module
    load time) rather than engine.config.DEFAULT_DB_URL: get_engine() is
    @st.cache_resource-wrapped with no arguments, so once it's been called
    anywhere in this process the cached Engine is fixed regardless of what
    DEFAULT_DB_URL says afterwards — patching the function itself is what
    actually keeps every test off the real data/hypocxe.db.
    """
    test_engine = init_db("sqlite://")
    monkeypatch.setattr(components.db, "get_engine", lambda: test_engine)
    at = AppTest.from_file(PAGE_PATH)
    at.run(timeout=30)
    return at


def _preview_button(at):
    return next(b for b in at.button if b.label == "Preview")


def _upload(at, files: list[tuple[str, bytes]]) -> None:
    at.file_uploader[0].set_value([(name, content, XLSX_MIME) for name, content in files])
    at.run(timeout=30)


def _synthetic_goal_workbook_bytes() -> bytes:
    """A minimal, valid single-sheet goal workbook with no Plan column and
    a sheet name that gives no plan-type signal either — detection for
    this file comes from its filename suffix alone, same as a real
    Format A export with an unhelpful sheet name would."""
    import io

    import pandas as pd

    buffer = io.BytesIO()
    df = pd.DataFrame(dict(
        Priority=[1], ROI=["Bladder"], Goal=["D0.03cc <= 4725 cGy"], GoalType=["DoseAtVolume"],
        Criteria=["AtMost"], AcceptanceLevel=[4725.0], ParameterValue=[0.03],
        AchievedValue=[4700.0], Status=["PASS"],
    ))
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Sheet1", index=False)
    return buffer.getvalue()


def _fill_required_fields(at, pt_no: str) -> None:
    for ti in at.text_input:
        if ti.label == "Patient no.":
            ti.set_value(pt_no)
        elif ti.label in ("Treatment room", "Medical physicist 1", "Medical physicist 2",
                          "Radiation oncologist"):
            ti.set_value("x")
    at.run(timeout=30)


# --------------------------------------------------------------------------- #
# Unambiguous 3-file upload
# --------------------------------------------------------------------------- #


def test_unambiguous_three_file_upload_shows_detected_summary(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("1clinical_goals_90000001_case.xlsx", _read("1clinical_goals_90000001_case.xlsx")),
    ])
    assert list(app.selectbox) == []  # nothing ambiguous -> no resolution UI at all
    assert list(app.warning) == []
    assert list(app.error) == []
    captions = [c.value for c in app.caption]
    assert ("Detected: Manual (1clinical_goals_90000001_Manual.xlsx), "
           "Auto (1clinical_goals_90000001_Auto.xlsx), "
           "Auto+Manual (1clinical_goals_90000001_case.xlsx)") in captions
    assert _preview_button(app).disabled is False


def test_unambiguous_three_file_upload_preview_produces_correct_plan_types(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("1clinical_goals_90000001_case.xlsx", _read("1clinical_goals_90000001_case.xlsx")),
    ])
    _fill_required_fields(app, "PtUnambiguous")

    _preview_button(app).click()
    app.run(timeout=30)

    assert list(app.exception) == []
    assert list(app.error) == []
    preview = app.session_state["new_case_preview"]
    assert {pf.plan_type for pf in preview["plan_frames"]} == {
        PlanType.MANUAL, PlanType.AUTO, PlanType.AUTO_MANUAL,
    }
    assert preview["form"].pt_no == "PtUnambiguous"
    assert preview["form"].hn == "90000001"  # resolved from the filenames


# --------------------------------------------------------------------------- #
# 2-file straight-pass upload
# --------------------------------------------------------------------------- #


def test_two_file_upload_is_a_straight_pass_candidate(app):
    """Manual + Auto only, no Auto+Manual file — must behave exactly like
    today: Preview enabled, Auto+Manual simply absent from the summary."""
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    assert list(app.selectbox) == []
    captions = [c.value for c in app.caption]
    detected = next(c for c in captions if c.startswith("Detected:"))
    assert detected == ("Detected: Manual (1clinical_goals_90000001_Manual.xlsx), "
                        "Auto (1clinical_goals_90000001_Auto.xlsx)")
    assert "Auto+Manual" not in detected
    assert _preview_button(app).disabled is False


def test_two_file_upload_preview_has_no_auto_manual_plan_frame(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    _fill_required_fields(app, "PtStraightPass")

    _preview_button(app).click()
    app.run(timeout=30)

    assert list(app.error) == []
    preview = app.session_state["new_case_preview"]
    assert {pf.plan_type for pf in preview["plan_frames"]} == {PlanType.MANUAL, PlanType.AUTO}


# --------------------------------------------------------------------------- #
# Conflicting pair resolved via selectbox
# --------------------------------------------------------------------------- #


def test_conflicting_pair_blocks_preview_until_resolved(app):
    """Two files whose Plan column/sheet name/filename all say 'Manual' —
    don't guess which slot each belongs in, ask."""
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
    ])
    assert len(app.selectbox) == 2
    assert all(s.value == "Manual" for s in app.selectbox)  # both default to their shared guess
    assert len(app.warning) == 1
    assert "Couldn't confidently place 2 files" in app.warning[0].value
    assert len(app.error) == 1
    assert ("Manual: 1clinical_goals_90000001_Manual.xlsx, 3clinical_goals_90000003_Manual.xlsx"
           in app.error[0].value)
    assert _preview_button(app).disabled is True


def test_conflicting_pair_resolved_by_reassigning_one_file(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
    ])
    boxes = list(app.selectbox)
    boxes[1].set_value("Auto")
    app.run(timeout=30)

    assert list(app.error) == []
    captions = [c.value for c in app.caption]
    assert ("Detected: Manual (1clinical_goals_90000001_Manual.xlsx), "
           "Auto (3clinical_goals_90000003_Manual.xlsx)") in captions
    assert _preview_button(app).disabled is False


def test_conflicting_pair_resolved_by_skipping_one_file(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("3clinical_goals_90000003_Manual.xlsx", _read("3clinical_goals_90000003_Manual.xlsx")),
    ])
    boxes = list(app.selectbox)
    boxes[1].set_value("Skip")
    app.run(timeout=30)

    # No more duplicate-assignment error, but Auto is now missing entirely
    # (both files were Manual candidates) -- Preview stays disabled, for a
    # different reason than the conflict.
    assert list(app.error) == []
    assert _preview_button(app).disabled is True
    captions = [c.value for c in app.caption]
    assert "Both a Manual and an Auto plan file are required (Auto+Manual is optional)." in captions


def test_conflicting_pair_resolved_preview_uses_the_chosen_types(app):
    """Two synthetic files sharing an HN (so the unrelated HN safeguard
    doesn't interfere) but both suffixed '_Manual' — a real conflict, since
    neither has a Plan column or a sheet name to disambiguate them."""
    content = _synthetic_goal_workbook_bytes()
    _upload(app, [
        ("caseA_90000001_Manual.xlsx", content),
        ("caseB_90000001_Manual.xlsx", content),
    ])
    boxes = list(app.selectbox)
    boxes[1].set_value("Auto")
    app.run(timeout=30)
    _fill_required_fields(app, "PtConflictResolved")

    _preview_button(app).click()
    app.run(timeout=30)

    assert list(app.error) == []
    preview = app.session_state["new_case_preview"]
    by_type = {pf.plan_type: pf.source_filename for pf in preview["plan_frames"]}
    assert by_type[PlanType.MANUAL] == "caseA_90000001_Manual.xlsx"
    assert by_type[PlanType.AUTO] == "caseB_90000001_Manual.xlsx"


# --------------------------------------------------------------------------- #
# Unmatchable file resolved via selectbox
# --------------------------------------------------------------------------- #


def test_unmatchable_file_shown_with_selectbox_not_silently_dropped(app):
    """The combined Format B fixture has no Plan column, and neither its
    (truncated) sheet name nor its filename contain a Manual/Auto/
    Auto+Manual token — it must show up for the user to assign, not
    vanish from the case."""
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("6clinical_goals_90000006_combined.xlsx", _read("6clinical_goals_90000006_combined.xlsx")),
    ])
    assert len(app.selectbox) == 1
    assert app.selectbox[0].value == "— choose —"  # no guess to default to
    assert any("6clinical_goals_90000006_combined.xlsx" in m.value
              and "couldn't detect a plan type" in m.value for m in app.markdown)
    assert _preview_button(app).disabled is True


def test_unmatchable_file_resolved_by_choosing_a_type(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("6clinical_goals_90000006_combined.xlsx", _read("6clinical_goals_90000006_combined.xlsx")),
    ])
    app.selectbox[0].set_value("Auto+Manual")
    app.run(timeout=30)

    assert _preview_button(app).disabled is False
    captions = [c.value for c in app.caption]
    assert ("Detected: Manual (1clinical_goals_90000001_Manual.xlsx), "
           "Auto (1clinical_goals_90000001_Auto.xlsx), "
           "Auto+Manual (6clinical_goals_90000006_combined.xlsx)") in captions


def test_unmatchable_file_can_be_skipped(app):
    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
        ("6clinical_goals_90000006_combined.xlsx", _read("6clinical_goals_90000006_combined.xlsx")),
    ])
    app.selectbox[0].set_value("Skip")
    app.run(timeout=30)

    assert _preview_button(app).disabled is False  # Manual+Auto still present without it
    captions = [c.value for c in app.caption]
    assert ("Detected: Manual (1clinical_goals_90000001_Manual.xlsx), "
           "Auto (1clinical_goals_90000001_Auto.xlsx)") in captions


# --------------------------------------------------------------------------- #
# Existing checks, kept unchanged — light regression coverage now that the
# AppTest harness exists (the substantive behavior is engine-level tested
# elsewhere: tests/test_parser.py for the HN safeguard, tests/test_storage.py
# for save_case's duplicate pt_no rejection)
# --------------------------------------------------------------------------- #


def test_hn_mismatch_between_files_still_blocks_preview(app):
    """4clinical_goals_90000004_Manual.xlsx is the fixture built to
    exercise the HN-mismatch rejection when paired against a deliberately
    wrong form HN."""
    _upload(app, [
        ("4clinical_goals_90000004_Manual.xlsx", _read("4clinical_goals_90000004_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    _fill_required_fields(app, "PtHnMismatch")

    _preview_button(app).click()
    app.run(timeout=30)

    assert "new_case_preview" not in app.session_state
    assert any("HN safeguard" in e.value for e in app.error)


def test_duplicate_pt_no_without_overwrite_still_blocks_preview(app):
    from engine.schemas import DoseRegimen, FormInput
    from engine.storage import save_case

    # components.db.get_engine is monkeypatched (no-arg) by the `app`
    # fixture, so calling it again here returns that same in-memory engine.
    test_engine = components.db.get_engine()
    save_case(test_engine, FormInput(
        hn="90000009", pt_no="PtDup", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
        fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C",
    ), [])

    _upload(app, [
        ("1clinical_goals_90000001_Manual.xlsx", _read("1clinical_goals_90000001_Manual.xlsx")),
        ("1clinical_goals_90000001_Auto.xlsx", _read("1clinical_goals_90000001_Auto.xlsx")),
    ])
    _fill_required_fields(app, "PtDup")

    _preview_button(app).click()
    app.run(timeout=30)

    assert "new_case_preview" not in app.session_state
    assert any("already exists" in e.value for e in app.error)
