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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence

from sqlalchemy import Column
from sqlalchemy import Enum as SAEnum
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select
import pandas as pd

from engine.config import DEFAULT_DB_URL
from engine.schemas import (
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
    "AnalysisRun",
    "init_db",
    "save_case",
    "load_analysis_frame",
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
    priority: int
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


class AnalysisRun(SQLModel, table=True):
    __tablename__ = "analysis_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=_utcnow)
    engine_version: str
    criteria_version: str
    settings_json: str


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


def save_case(engine: Engine, form: FormInput, plan_frames: Sequence[PlanFrame]) -> int:
    """Persist one case (a patient plus one or more plans and their goal
    results) as a single transaction. Returns the new patient's id.

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
            )
            session.add(patient)
            session.flush()  # assigns patient.id without ending the transaction

            for frame in plan_frames:
                plan = Plan(
                    patient_id=patient.id,
                    plan_type=frame.plan_type,
                    planning_time_min=frame.planning_time_min,
                    is_straight_pass=frame.is_straight_pass,
                    source_filename=frame.source_filename,
                    file_sha256=frame.file_sha256,
                )
                session.add(plan)
                session.flush()  # assigns plan.id

                for goal in frame.goals:
                    session.add(GoalResult(plan_id=plan.id, **goal.model_dump()))

            session.commit()
            return patient.id
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
    "goal_key", "priority", "roi_raw", "roi", "goal_text", "goal_type",
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
