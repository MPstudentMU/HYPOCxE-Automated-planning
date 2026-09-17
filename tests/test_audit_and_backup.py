"""Tests for engine/storage.py's record_analysis_run/recent_activity/
ensure_daily_backup and engine/corrections.py's load_all_corrections —
the audit-trail and backup support behind Module 9 (Registry Export) and
the sidebar activity log."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from engine import corrections as C
from engine import parser as P
from engine.schemas import CorrectionField, CorrectionSource, CorrectionStatus, DoseRegimen, FormInput
from engine.storage import (
    ensure_daily_backup,
    init_db,
    load_analysis_frame,
    record_analysis_run,
    recent_activity,
    save_case,
)

FIXTURES = "tests/fixtures/pilot"


def _form(**overrides) -> FormInput:
    defaults = dict(hn="90000001", pt_no="Pt1", dose_regimen=DoseRegimen.HYPO, rx_cgy=4400,
                    fractions=20, sib_boost=False, tx_room="R1", mp1="A", mp2="B", ro="C")
    defaults.update(overrides)
    return FormInput(**defaults)


@pytest.fixture
def engine():
    return init_db("sqlite://")


# --------------------------------------------------------------------------- #
# record_analysis_run
# --------------------------------------------------------------------------- #


def test_record_analysis_run_returns_an_id(engine):
    run_id = record_analysis_run(engine, engine_version="2.0", criteria_version="v0",
                                 settings_json="{}")
    assert isinstance(run_id, int)


def test_record_analysis_run_ids_increment(engine):
    id1 = record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")
    id2 = record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")
    assert id2 > id1


def test_record_analysis_run_stamps_entered_by(engine):
    from sqlmodel import Session
    from engine.storage import AnalysisRun

    run_id = record_analysis_run(engine, engine_version="2.0", criteria_version="v0",
                                 settings_json="{}", entered_by="Dr. Somchai")
    with Session(engine) as session:
        assert session.get(AnalysisRun, run_id).entered_by == "Dr. Somchai"


def test_record_analysis_run_entered_by_defaults_to_none(engine):
    from sqlmodel import Session
    from engine.storage import AnalysisRun

    run_id = record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")
    with Session(engine) as session:
        assert session.get(AnalysisRun, run_id).entered_by is None


# --------------------------------------------------------------------------- #
# recent_activity
# --------------------------------------------------------------------------- #


def test_recent_activity_empty_db(engine):
    activity = recent_activity(engine)
    assert activity.empty
    assert list(activity.columns) == ["timestamp", "action", "detail"]


def test_recent_activity_includes_new_case(engine):
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)

    activity = recent_activity(engine)
    new_case_rows = activity[activity["action"] == "New case"]
    assert len(new_case_rows) == 1
    assert new_case_rows.iloc[0]["detail"] == "Pt1"


def test_recent_activity_includes_correction_with_no_hn(engine):
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)

    df = load_analysis_frame(engine)
    bone_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    C.record_correction(engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=800.0)

    activity = recent_activity(engine)
    correction_rows = activity[activity["action"] == "Correction"]
    assert len(correction_rows) == 1
    detail = correction_rows.iloc[0]["detail"]
    assert "Pt1" in detail and "Dr. Test" in detail
    assert "90000001" not in detail  # the HN, must never appear


def test_recent_activity_includes_export(engine):
    record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")
    activity = recent_activity(engine)
    assert (activity["action"] == "Export").sum() == 1


def test_recent_activity_sorted_most_recent_first(engine):
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)  # earlier
    record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")  # later

    activity = recent_activity(engine)
    assert activity.iloc[0]["action"] == "Export"  # most recent first


def test_recent_activity_respects_limit(engine):
    for i in range(5):
        record_analysis_run(engine, engine_version="2.0", criteria_version="v0", settings_json="{}")
    activity = recent_activity(engine, limit=3)
    assert len(activity) == 3


def test_recent_activity_survives_superseded_correction(engine):
    """A re-upload deletes the old goal_results row a correction targeted
    (after flagging it superseded_by_upload) — the sidebar log must still
    show the correction, just without a resolvable pt_no."""
    from engine.storage import replace_plan, find_patient_by_pt_no

    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)
    df = load_analysis_frame(engine)
    bone_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    C.record_correction(engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured", corrected_by="Dr. Test", corrected_value_display=800.0)

    reparsed = P.parse_case_files([f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx"], _form())
    patient = find_patient_by_pt_no(engine, "Pt1")
    replace_plan(engine, patient.id, reparsed.plan_frames[0])  # re-upload -> supersedes

    activity = recent_activity(engine)
    correction_rows = activity[activity["action"] == "Correction"]
    assert len(correction_rows) == 1
    detail = correction_rows.iloc[0]["detail"]
    assert "Dr. Test" in detail
    assert "90000001" not in detail


def test_recent_activity_never_carries_hn_column(engine):
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)
    activity = recent_activity(engine)
    assert "hn" not in activity.columns


# --------------------------------------------------------------------------- #
# ensure_daily_backup
# --------------------------------------------------------------------------- #


def test_ensure_daily_backup_noop_for_in_memory_db():
    assert ensure_daily_backup("sqlite://") is None


def test_ensure_daily_backup_noop_when_file_does_not_exist_yet(tmp_path):
    url = f"sqlite:///{tmp_path}/does_not_exist.db"
    assert ensure_daily_backup(url) is None


def test_ensure_daily_backup_creates_dated_copy(tmp_path):
    db_path = tmp_path / "hypocxe.db"
    init_db(f"sqlite:///{db_path}")  # creates the file
    assert db_path.exists()

    backup_dir = tmp_path / "backups"
    result = ensure_daily_backup(f"sqlite:///{db_path}", backup_dir=backup_dir)

    assert result is not None
    assert result.exists()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    assert result.name == f"hypocxe_{today}.db"
    assert result.read_bytes() == db_path.read_bytes()


def test_ensure_daily_backup_second_call_same_day_is_noop(tmp_path):
    db_path = tmp_path / "hypocxe.db"
    init_db(f"sqlite:///{db_path}")
    backup_dir = tmp_path / "backups"

    first = ensure_daily_backup(f"sqlite:///{db_path}", backup_dir=backup_dir)
    second = ensure_daily_backup(f"sqlite:///{db_path}", backup_dir=backup_dir)

    assert first is not None
    assert second is None
    assert len(list(backup_dir.iterdir())) == 1  # not duplicated


def test_ensure_daily_backup_creates_backup_dir_if_missing(tmp_path):
    db_path = tmp_path / "hypocxe.db"
    init_db(f"sqlite:///{db_path}")
    backup_dir = tmp_path / "nested" / "backups"
    assert not backup_dir.exists()

    ensure_daily_backup(f"sqlite:///{db_path}", backup_dir=backup_dir)
    assert backup_dir.exists()


# --------------------------------------------------------------------------- #
# load_all_corrections
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine_with_correction():
    engine = init_db("sqlite://")
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)
    df = load_analysis_frame(engine)
    bone_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    C.record_correction(engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.RAYSTATION_DVH,
                        reason="measured from DVH", corrected_by="Dr. Test",
                        corrected_value_display=800.0)
    return engine, bone_id


def test_load_all_corrections_empty_db():
    engine = init_db("sqlite://")
    corrections = C.load_all_corrections(engine)
    assert corrections.empty
    assert "hn" not in corrections.columns


def test_load_all_corrections_returns_the_correction(engine_with_correction):
    engine, _ = engine_with_correction
    corrections = C.load_all_corrections(engine)
    assert len(corrections) == 1
    row = corrections.iloc[0]
    assert row["pt_no"] == "Pt1"
    assert row["plan_type"] == "Manual"
    assert row["roi"] == "Bone Marrow"
    assert row["field"] == "AchievedValue"
    assert row["status"] == "corrected"
    assert row["source"] == "RayStation DVH"
    assert row["reason"] == "measured from DVH"
    assert row["corrected_by"] == "Dr. Test"
    assert row["corrected_value"] == "800.0"
    assert row["superseded_by_upload"] == False  # noqa: E712


def test_load_all_corrections_never_carries_hn(engine_with_correction):
    engine, _ = engine_with_correction
    corrections = C.load_all_corrections(engine)
    assert "hn" not in corrections.columns
    assert not corrections.astype(str).isin(["90000001"]).any().any()


def test_load_all_corrections_includes_superseded_rows(engine_with_correction):
    """A superseded correction stays in the audit trail — completeness,
    not just what's currently active."""
    from engine.storage import replace_plan, find_patient_by_pt_no

    engine, _ = engine_with_correction
    reparsed = P.parse_case_files([f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx"], _form())
    patient = find_patient_by_pt_no(engine, "Pt1")
    replace_plan(engine, patient.id, reparsed.plan_frames[0])  # re-upload -> supersedes

    corrections = C.load_all_corrections(engine)
    assert len(corrections) == 1
    assert corrections.iloc[0]["superseded_by_upload"] == True  # noqa: E712


def test_load_all_corrections_sorted_most_recent_first():
    engine = init_db("sqlite://")
    parsed = P.parse_case_files([
        f"{FIXTURES}/1clinical_goals_90000001_Manual.xlsx",
        f"{FIXTURES}/1clinical_goals_90000001_Auto.xlsx",
    ], _form())
    save_case(engine, _form(), parsed.plan_frames)
    df = load_analysis_frame(engine)
    bone_id = int(df.loc[df["roi"] == "Bone Marrow", "goal_id"].iloc[0])
    rectum_id = int(df.loc[df["roi_raw"] == "Rectum_new", "goal_id"].iloc[0])

    C.record_correction(engine, goal_id=bone_id, field=CorrectionField.ACHIEVED_VALUE,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.OTHER,
                        reason="first", corrected_by="A", corrected_value_display=800.0)
    C.record_correction(engine, goal_id=rectum_id, field=CorrectionField.PRIORITY,
                        status=CorrectionStatus.CORRECTED, source=CorrectionSource.PROTOCOL,
                        reason="second", corrected_by="B", corrected_value_display=2)

    corrections = C.load_all_corrections(engine)
    assert corrections.iloc[0]["reason"] == "second"  # most recent first
    assert corrections.iloc[1]["reason"] == "first"
