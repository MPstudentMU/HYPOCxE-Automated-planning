"""Pydantic models for data entering the database.

These are the validation boundary between the outside world (the intake
form, parsed export files) and engine/storage.py. Nothing here talks to the
database; engine/storage.py converts validated instances of these models
into ORM rows.

See CLAUDE.md — plan types are exactly "Manual", "Auto", "Auto+Manual"
(rule 5), and HN must never reach analysis output (rule 4); FormInput is one
of the few places HN is legitimately collected at all.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- #
# Enumerations shared between the Pydantic models here and the ORM models in
# engine/storage.py.
# --------------------------------------------------------------------------- #


class DoseRegimen(str, Enum):
    HYPO = "Hypo"
    CONV = "Conv"


class PlanType(str, Enum):
    """Exactly the three plan types in CLAUDE.md rule 5. Member names avoid
    the '+' character (not legal in a Python identifier); member *values*
    are the exact strings used everywhere else in the project — see
    engine/priority_filter.py:PLAN_ORDER."""

    MANUAL = "Manual"
    AUTO = "Auto"
    AUTO_MANUAL = "Auto+Manual"


class CriteriaDirection(str, Enum):
    AT_MOST = "AtMost"
    AT_LEAST = "AtLeast"


class GoalStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class StructureClass(str, Enum):
    TARGET = "TARGET"
    OAR = "OAR"


# Priority levels a goal template can carry. Kept local (rather than imported
# from engine/priority_filter.py) so this low-level schema module has no
# dependency on the analysis layer; must stay in sync with
# engine/priority_filter.py:PRIORITY_LEVELS.
VALID_PRIORITIES = (1, 2, 3)

# Which FormInput field holds the planning time for a given plan type.
FORM_TIME_FIELD: dict[PlanType, str] = {
    PlanType.MANUAL: "time_manual",
    PlanType.AUTO: "time_auto",
    PlanType.AUTO_MANUAL: "time_automanual",
}


# --------------------------------------------------------------------------- #
# Intake form
# --------------------------------------------------------------------------- #


class FormInput(BaseModel):
    """One completed intake form for a case (Module 0 — New Case).

    rx_cgy and fractions aren't in the field list the platform spec gave for
    this model, but engine.storage.Patient requires both (NOT NULL columns)
    and nothing else in the intake flow is positioned to supply them, so
    they're collected here too rather than guessed at save time — see
    CLAUDE.md rule 1 on not inventing values the manual should define.

    hn is optional here (unlike engine.storage.Patient.hn, which is
    NOT NULL): engine/parser.py's HN safeguard allows leaving it blank on
    the form and auto-filling it from the uploaded files' filenames when
    they all agree. Resolve it to a definite string (form-entered or
    filename-derived) before constructing the Patient row.
    """

    model_config = ConfigDict(use_enum_values=False)

    hn: Optional[str] = Field(default=None, min_length=1)
    pt_no: str = Field(min_length=1)
    dose_regimen: DoseRegimen
    rx_cgy: float = Field(gt=0)
    fractions: int = Field(gt=0)
    sib_boost: bool
    tx_room: str = Field(min_length=1)
    mp1: str = Field(min_length=1)
    mp2: str = Field(min_length=1)
    ro: str = Field(min_length=1)

    time_manual: Optional[float] = Field(default=None, gt=0)
    time_auto: Optional[float] = Field(default=None, gt=0)
    time_automanual: Optional[float] = Field(default=None, gt=0)

    def time_for(self, plan_type: PlanType) -> Optional[float]:
        """Planning time (minutes) the form recorded for this plan type."""
        return getattr(self, FORM_TIME_FIELD[plan_type])


# --------------------------------------------------------------------------- #
# Goal-level result
# --------------------------------------------------------------------------- #


class GoalRow(BaseModel):
    """One evaluated goal for one plan (a row of engine.storage.GoalResult,
    minus the id/plan_id assigned at save time).

    acceptance_level was originally modeled here as an optional qualitative
    label — a guess flagged as unconfirmed pending the manual. Real pilot
    exports (RayStation clinical-goal sheets) show it is always a required
    numeric threshold compared directly against achieved_value (e.g. Criteria
    AtMost + AcceptanceLevel 4820 + AchievedValue 4802.6 -> PASS), so it's
    corrected to float here. See engine/parser.py §2.2 for how it's used.
    """

    model_config = ConfigDict(use_enum_values=False)

    goal_key: str = Field(min_length=1)
    priority: int
    roi_raw: str = Field(min_length=1)
    roi: str = Field(min_length=1)
    goal_text: str = Field(min_length=1)
    goal_type: str = Field(min_length=1)
    criteria: CriteriaDirection
    acceptance_level: float
    parameter_value: float
    achieved_value: Optional[float] = None
    status: Optional[GoalStatus] = None
    evaluable: bool
    structure_class: StructureClass

    @model_validator(mode="after")
    def _check_priority(self) -> "GoalRow":
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {VALID_PRIORITIES}, got {self.priority!r}"
            )
        return self


# --------------------------------------------------------------------------- #
# Per-plan bundle passed to storage.save_case
# --------------------------------------------------------------------------- #


class PlanFrame(BaseModel):
    """Everything needed to persist one plan for one case: the plan-level
    metadata (mostly sourced from the parsed export file) plus its scored
    goals. Not part of the platform spec's named model list, but save_case's
    second argument needs some container for this, and it's simplest as a
    validated Pydantic model alongside FormInput/GoalRow rather than an
    untyped dict."""

    model_config = ConfigDict(use_enum_values=False)

    plan_type: PlanType
    planning_time_min: Optional[float] = Field(default=None, gt=0)
    is_straight_pass: Optional[bool] = None
    source_filename: Optional[str] = None
    file_sha256: Optional[str] = None
    goals: list[GoalRow] = Field(default_factory=list)
