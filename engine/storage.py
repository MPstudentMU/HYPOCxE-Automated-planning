"""SQLModel schema and database access.

Tables
------
patients       one row per case intake (Module 0). Carries HN — never expose
               this table, or a query that includes its hn column, to
               anything downstream of load_analysis_frame(). See
               CLAUDE.md rule 4.
plans          one row per plan (Manual / Auto / Auto+Manual) for a patient.
goal_results   one row per evaluated goal for a plan.
analysis_runs  a log of analysis runs (engine/criteria version, settings),
               independent of any single patient or plan.

Only three functions here matter to callers outside this module:
init_db(), save_case(), load_analysis_frame(). Everything else is the ORM
model definitions.

Note: this file deliberately does NOT use
`from __future__ import annotations` (unlike the rest of engine/). SQLModel's
Relationship() resolves forward references like List["Plan"] by evaluating
real annotation objects at mapper-configure time; with postponed evaluation
those annotations are plain strings and SQLAlchemy fails to parse the
generic, raising "using a generic class as the argument to relationship()".
"""
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

from sqlalchemy import Column, delete
from sqlalchemy import Enum as SAEnum
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select
import pandas as pd

from engine.config import DEFAULT_DB_URL
from engine.schemas import (
    CorrectionField,
    CorrectionSource,
    CorrectionStatus,
    CriteriaDirection,
    DoseRegimen,
    FormInput,
    GoalStatus,
    PlanFrame,
    PlanType,
    StructureClass,
)

__all__ = [
    "Patient",
    "Plan",
    "GoalResult",
    "GoalCorrection",
    "AnalysisRun",
    "init_db",
    "save_case",
    "replace_plan",
    "find_patient_by_pt_no",
    "update_patient",
    "load_analysis_frame",
    "record_analysis_run",
    "recent_activity",
    "ensure_daily_backup",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum_column(enum_cls: type[Enum], *, nullable: bool = False) -> Column:
    """A SQLAlchemy Enum column that stores each member's *value* (e.g.
    "Auto+Manual"), not its Python identifier — so the database text matches
    CLAUDE.md rule 5 exactly and is readable without the ORM."""
    return Column(SAEnum(enum_cls, values_callable=lambda cls: [e.value for e in cls]), nullable=nullable)


# --------------------------------------------------------------------------- #
# ORM models
# --------------------------------------------------------------------------- #


class Patient(SQLModel, table=True):
    __tablename__ = "patients"

    id: Optional[int] = Field(default=None, primary_key=True)
    pt_no: str = Field(unique=True, index=True)
    hn: str
    dose_regimen: DoseRegimen = Field(sa_column=_enum_column(DoseRegimen))
    rx_cgy: float
    fractions: int
    sib_boost: bool
    tx_room: str
    mp1: str
    mp2: str
    ro: str
    created_at: datetime = Field(default_factory=_utcnow)
    # Who was logged in (the session's one-time "Entered by" name — see
    # engine.auth / app.py) when this case was saved. Session-only
    # attribution, not a user account; None for rows saved before this
    # column existed or outside the app (e.g. most tests).
    entered_by: Optional[str] = None

    plans: List["Plan"] = Relationship(back_populates="patient")


class Plan(SQLModel, table=True):
    __tablename__ = "plans"

    id: Optional[int] = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patients.id", index=True)
    plan_type: PlanType = Field(sa_column=_enum_column(PlanType))
    planning_time_min: Optional[float] = None
    is_straight_pass: Optional[bool] = None
    source_filename: Optional[str] = None
    file_sha256: Optional[str] = Field(default=None, index=True)

    patient: Optional[Patient] = Relationship(back_populates="plans")
    goal_results: List["GoalResult"] = Relationship(back_populates="plan")


class GoalResult(SQLModel, table=True):
    __tablename__ = "goal_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: int = Field(foreign_key="plans.id", index=True)
    goal_key: str = Field(index=True)
    # Nullable: RayStation's "no priority" sentinel (engine/parser.py) is
    # normalized to None and the row is kept, not dropped, so it can be
    # reviewed and corrected (engine/corrections.py) instead of silently
    # vanishing from goal_results.
    priority: Optional[int] = None
    roi_raw: str
    roi: str
    goal_text: str
    goal_type: str
    criteria: CriteriaDirection = Field(sa_column=_enum_column(CriteriaDirection))
    acceptance_level: float
    parameter_value: float
    achieved_value: Optional[float] = None
    status: Optional[GoalStatus] = Field(default=None, sa_column=_enum_column(GoalStatus, nullable=True))
    evaluable: bool
    structure_class: StructureClass = Field(sa_column=_enum_column(StructureClass))

    plan: Optional[Plan] = Relationship(back_populates="goal_results")
    corrections: List["GoalCorrection"] = Relationship(back_populates="goal")


class GoalCorrection(SQLModel, table=True):
    """A manual correction to one field of one goal_results row, per the
    Module 0/8 correction workflow. engine/corrections.py is the only code
    that should write here or read this to build a corrected analysis
    frame — see that module's docstring. goal_results itself is never
    modified; a correction is always an overlay applied at analysis time.
    """
    __tablename__ = "goal_corrections"

    id: Optional[int] = Field(default=None, primary_key=True)
    goal_id: int = Field(foreign_key="goal_results.id", index=True)
    field: CorrectionField = Field(sa_column=_enum_column(CorrectionField))
    # Stored as text regardless of the underlying field's real type (float
    # for AchievedValue, int for Priority) since one column has to hold
    # both; engine/corrections.py parses back to the right type on read.
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    unit_entered: Optional[str] = None
    status: CorrectionStatus = Field(sa_column=_enum_column(CorrectionStatus))
    source: CorrectionSource = Field(sa_column=_enum_column(CorrectionSource))
    reason: str
    corrected_by: str
    corrected_at: datetime = Field(default_factory=_utcnow)
    # True once a re-upload replaces the goal_results row this correction
    # targeted — the correction stays in the table (audit trail), it just
    # stops being applied. See engine.storage.replace_plan.
    superseded_by_upload: bool = False

    goal: Optional[GoalResult] = Relationship(back_populates="corrections")


class AnalysisRun(SQLModel, table=True):
    __tablename__ = "analysis_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow)
    engine_version: str
    criteria_version: str
    settings_json: str
    # The session's "Entered by" name at export time (see Patient.entered_by
    # above) — None for a run logged outside the app, or before this column
    # existed.
    entered_by: Optional[str] = None


# --------------------------------------------------------------------------- #
# init_db
# --------------------------------------------------------------------------- #


def _is_memory_url(url: str) -> bool:
    return url == "sqlite://" or ":memory:" in url


def init_db(url: str = DEFAULT_DB_URL) -> Engine:
    """Create the engine and all tables (idempotent — safe to call on an
    existing database; it only creates tables that don't exist yet).

    Pass "sqlite://" (or any URL containing ":memory:") for an in-memory
    database, e.g. in tests.
    """
    connect_args = {"check_same_thread": False}
    if _is_memory_url(url):
        engine = create_engine(url, connect_args=connect_args, poolclass=StaticPool)
    else:
        if url.startswith("sqlite:///"):
            db_path = Path(url[len("sqlite:///"):])
            db_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, connect_args=connect_args)
    SQLModel.metadata.create_all(engine)
    return engine


# --------------------------------------------------------------------------- #
# save_case
# --------------------------------------------------------------------------- #


def _upsert_plan(session: Session, patient_id: int, frame: PlanFrame) -> int:
    """Insert a plan for patient_id, or — if one of the same plan_type
    already exists (a re-upload) — replace its goal_results with the new
    frame's. Any still-active correction targeting a replaced goal is
    marked superseded_by_upload=True (kept for the audit trail, no longer
    applied) rather than deleted. Must run inside an open transaction;
    raises and leaves cleanup to the caller on failure. Returns the plan id.
    """
    existing = session.exec(
        select(Plan).where(Plan.patient_id == patient_id, Plan.plan_type == frame.plan_type)
    ).first()

    if existing is None:
        plan = Plan(
            patient_id=patient_id,
            plan_type=frame.plan_type,
            planning_time_min=frame.planning_time_min,
            is_straight_pass=frame.is_straight_pass,
            source_filename=frame.source_filename,
            file_sha256=frame.file_sha256,
        )
        session.add(plan)
        session.flush()
        plan_id = plan.id
    else:
        old_goal_ids = session.exec(
            select(GoalResult.id).where(GoalResult.plan_id == existing.id)
        ).all()
        if old_goal_ids:
            stale_corrections = session.exec(
                select(GoalCorrection)
                .where(GoalCorrection.goal_id.in_(old_goal_ids))
                .where(GoalCorrection.superseded_by_upload == False)  # noqa: E712
            ).all()
            for correction in stale_corrections:
                correction.superseded_by_upload = True
                session.add(correction)
            session.flush()  # write the supersede flags before the goals disappear

            # A bulk DELETE, not session.delete(obj): the ORM relationship
            # would otherwise try to null out goal_corrections.goal_id for
            # every row that references a deleted goal (its default
            # one-to-many cascade), which violates that column's NOT NULL
            # constraint — and would erase exactly the history we just
            # flagged superseded_by_upload to preserve.
            session.exec(delete(GoalResult).where(GoalResult.id.in_(old_goal_ids)))

        existing.planning_time_min = frame.planning_time_min
        existing.is_straight_pass = frame.is_straight_pass
        existing.source_filename = frame.source_filename
        existing.file_sha256 = frame.file_sha256
        session.add(existing)
        session.flush()
        plan_id = existing.id

    for goal in frame.goals:
        session.add(GoalResult(plan_id=plan_id, **goal.model_dump()))

    return plan_id


def save_case(engine: Engine, form: FormInput, plan_frames: Sequence[PlanFrame], *,
              entered_by: Optional[str] = None) -> int:
    """Persist one case (a patient plus one or more plans and their goal
    results) as a single transaction. Returns the new patient's id.

    `entered_by` is the app session's one-time "Entered by" name (see
    engine.auth / app.py) — stamped onto the new Patient row for
    traceability; None outside the app (most tests don't pass it).

    Raises whatever the underlying database raises (e.g. IntegrityError on a
    duplicate pt_no) after rolling the whole transaction back — no partial
    patient/plan/goal_result rows are left behind on failure.
    """
    with Session(engine) as session:
        try:
            patient = Patient(
                pt_no=form.pt_no,
                hn=form.hn,
                dose_regimen=form.dose_regimen,
                rx_cgy=form.rx_cgy,
                fractions=form.fractions,
                sib_boost=form.sib_boost,
                tx_room=form.tx_room,
                mp1=form.mp1,
                mp2=form.mp2,
                ro=form.ro,
                entered_by=entered_by,
            )
            session.add(patient)
            session.flush()  # assigns patient.id without ending the transaction

            for frame in plan_frames:
                _upsert_plan(session, patient.id, frame)

            session.commit()
            return patient.id
        except Exception:
            session.rollback()
            raise


def replace_plan(engine: Engine, patient_id: int, plan_frame: PlanFrame) -> int:
    """Add a plan to an existing patient, or replace it (a re-upload) if
    one of that plan_type already exists — see _upsert_plan. Returns the
    plan id.
    """
    with Session(engine) as session:
        try:
            plan_id = _upsert_plan(session, patient_id, plan_frame)
            session.commit()
            return plan_id
        except Exception:
            session.rollback()
            raise


def find_patient_by_pt_no(engine: Engine, pt_no: str) -> Optional[Patient]:
    """Look up an existing patient — used by the intake page to decide
    whether an upload is a new case (save_case) or a re-upload for an
    existing one (update_patient + replace_plan per plan)."""
    with Session(engine) as session:
        return session.exec(select(Patient).where(Patient.pt_no == pt_no)).first()


def update_patient(engine: Engine, patient_id: int, form: FormInput, *,
                    entered_by: Optional[str] = None) -> None:
    """Update an existing patient's intake fields in place (a re-upload
    that also corrects the case's metadata). Does not touch its plans —
    see replace_plan for that.

    `entered_by` (the app session's "Entered by" name — see save_case)
    overwrites the stored attribution only when given; None leaves whoever
    was recorded on the original save/last update untouched, rather than
    blanking it out for a caller that doesn't know about it.
    """
    with Session(engine) as session:
        try:
            patient = session.get(Patient, patient_id)
            if patient is None:
                raise ValueError(f"No such patient (id={patient_id})")
            patient.hn = form.hn
            patient.dose_regimen = form.dose_regimen
            patient.rx_cgy = form.rx_cgy
            patient.fractions = form.fractions
            patient.sib_boost = form.sib_boost
            patient.tx_room = form.tx_room
            patient.mp1 = form.mp1
            patient.mp2 = form.mp2
            patient.ro = form.ro
            if entered_by is not None:
                patient.entered_by = entered_by
            session.add(patient)
            session.commit()
        except Exception:
            session.rollback()
            raise


# --------------------------------------------------------------------------- #
# load_analysis_frame
# --------------------------------------------------------------------------- #

_ANALYSIS_COLUMNS = [
    "pt_no", "dose_regimen", "rx_cgy", "fractions", "sib_boost",
    "tx_room", "mp1", "mp2", "ro",
    "plan_id", "plan_type", "planning_time_min", "is_straight_pass", "source_filename",
    "goal_id", "goal_key", "priority", "roi_raw", "roi", "goal_text", "goal_type",
    "criteria", "acceptance_level", "parameter_value", "achieved_value",
    "status", "evaluable", "structure_class",
]

# Columns above that hold an Enum from engine.schemas and must be unwrapped
# to their plain string value before the frame leaves this module.
_ENUM_COLUMNS = ["dose_regimen", "plan_type", "criteria", "status", "structure_class"]


def load_analysis_frame(engine: Engine) -> pd.DataFrame:
    """One long DataFrame: patients joined to plans joined to goal_results —
    one row per goal per plan per patient. HN is deliberately excluded; the
    only patient identifier in the result is pt_no. See CLAUDE.md rule 4.

    analysis_runs is not part of this join: it logs analysis runs, not
    per-patient data, and has no foreign key into patients/plans/goal_results.
    """
    statement = (
        select(
            Patient.pt_no,
            Patient.dose_regimen,
            Patient.rx_cgy,
            Patient.fractions,
            Patient.sib_boost,
            Patient.tx_room,
            Patient.mp1,
            Patient.mp2,
            Patient.ro,
            Plan.id.label("plan_id"),
            Plan.plan_type,
            Plan.planning_time_min,
            Plan.is_straight_pass,
            Plan.source_filename,
            GoalResult.id.label("goal_id"),
            GoalResult.goal_key,
            GoalResult.priority,
            GoalResult.roi_raw,
            GoalResult.roi,
            GoalResult.goal_text,
            GoalResult.goal_type,
            GoalResult.criteria,
            GoalResult.acceptance_level,
            GoalResult.parameter_value,
            GoalResult.achieved_value,
            GoalResult.status,
            GoalResult.evaluable,
            GoalResult.structure_class,
        )
        .select_from(Patient)
        .join(Plan, Plan.patient_id == Patient.id)
        .join(GoalResult, GoalResult.plan_id == Plan.id)
    )
    with Session(engine) as session:
        rows = session.exec(statement).all()

    df = pd.DataFrame(rows, columns=_ANALYSIS_COLUMNS)

    for col in _ENUM_COLUMNS:
        df[col] = df[col].map(lambda v: v.value if isinstance(v, Enum) else v)

    assert "hn" not in df.columns, "HN must never enter the analysis frame — CLAUDE.md rule 4"
    return df


# --------------------------------------------------------------------------- #
# record_analysis_run / recent_activity / ensure_daily_backup
# --------------------------------------------------------------------------- #


def record_analysis_run(engine: Engine, *, engine_version: str, criteria_version: str,
                        settings_json: str, entered_by: Optional[str] = None) -> int:
    """Log one row to analysis_runs. engine/export.py calls this every
    time it builds an export workbook, so every export is traceable to
    exactly the engine/criteria version and settings that produced it —
    plus, now, who was logged in (entered_by) when they built it."""
    with Session(engine) as session:
        try:
            run = AnalysisRun(engine_version=engine_version, criteria_version=criteria_version,
                              settings_json=settings_json, entered_by=entered_by)
            session.add(run)
            session.commit()
            session.refresh(run)
            return run.id
        except Exception:
            session.rollback()
            raise


def recent_activity(engine: Engine, *, limit: int = 20) -> pd.DataFrame:
    """A unified, HN-free audit log for the sidebar: new-case saves
    (Patient.created_at), corrections (GoalCorrection.corrected_at), and
    analysis exports (AnalysisRun.timestamp) — most recent first. Derived
    from tables that already record these events (no separate audit
    table); patients are identified by pt_no only, never hn.
    """
    # LEFT OUTER, not inner: a re-upload (replace_plan) deletes the old
    # goal_results row a superseded correction targeted, after flagging
    # it superseded_by_upload — an inner join would silently drop that
    # correction from the log instead of just losing its pt_no.
    with Session(engine) as session:
        patients = session.exec(select(Patient.pt_no, Patient.created_at)).all()
        corrections = session.exec(
            select(GoalCorrection.corrected_at, GoalCorrection.corrected_by,
                  GoalCorrection.field, Patient.pt_no)
            .select_from(GoalCorrection)
            .join(GoalResult, GoalResult.id == GoalCorrection.goal_id, isouter=True)
            .join(Plan, Plan.id == GoalResult.plan_id, isouter=True)
            .join(Patient, Patient.id == Plan.patient_id, isouter=True)
        ).all()
        runs = session.exec(select(AnalysisRun.timestamp)).all()

    rows = []
    for pt_no, created_at in patients:
        rows.append(dict(timestamp=created_at, action="New case", detail=pt_no))
    for corrected_at, corrected_by, field, pt_no in corrections:
        field_label = field.value if isinstance(field, Enum) else field
        pt_no = pt_no or "a superseded goal"
        rows.append(dict(timestamp=corrected_at, action="Correction",
                         detail=f"{pt_no} · {field_label} · by {corrected_by}"))
    for ts in runs:
        rows.append(dict(timestamp=ts, action="Export", detail="Analysis workbook"))

    df = pd.DataFrame(rows, columns=["timestamp", "action", "detail"])
    if df.empty:
        return df
    return df.sort_values("timestamp", ascending=False).head(limit).reset_index(drop=True)


def ensure_daily_backup(url: str = DEFAULT_DB_URL, *, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """Copy the live SQLite file into data/backups/ once per calendar day
    (UTC), if today's backup doesn't already exist. A no-op for an
    in-memory database (nothing to copy) or before the db file exists yet
    (nothing to back up yet). This is a same-machine safety net against
    an accidental delete/corruption during a session, not a substitute
    for a real off-machine backup — see the README's local-network/PDPA
    section.
    """
    if _is_memory_url(url) or not url.startswith("sqlite:///"):
        return None
    db_path = Path(url[len("sqlite:///"):])
    if not db_path.exists():
        return None

    target_dir = backup_dir or (db_path.parent / "backups")
    target_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date().isoformat()
    backup_path = target_dir / f"{db_path.stem}_{today}{db_path.suffix}"
    if backup_path.exists():
        return None  # already have today's backup

    shutil.copy2(db_path, backup_path)
    return backup_path
