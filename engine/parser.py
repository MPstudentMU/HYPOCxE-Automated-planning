"""RayStation clinical-goal export parser.

Implements the intake-parsing rules from docs/analysis_manual_th_v2.md §2.2
and §2.4: Format A (one file per plan, one row per clinical goal — the
standard RayStation export) and Format B (one combined workbook holding
several plans, either as several Format-A-shaped sheets or as one sheet
with wide Achieved_<Plan> / Status_<Plan> columns).

NOTE ON PROVENANCE — this module was originally built before the manual
was in this repo, cross-checked only against the *structure* of real pilot
RayStation exports found on disk (column names, sheet-name truncation, the
Pt5 63-duplicate-row fact) without copying those files' content (which
carry real hospital numbers) — see tests/fixtures/pilot/ for the
de-identified synthetic fixtures used instead. Now that the manual is
present, it has been reconciled against §2.4's ROI harmonisation table
(see canonical_roi) per CLAUDE.md rule 1; the rest of this module's
behavior was already consistent with it.

Pipeline, per file:
  1. read every sheet
  2. normalize headers (strip whitespace, map known spellings)
  3. detect layout: Format A (single AchievedValue column) or Format B
     (wide Achieved_<Plan> columns) -> melt Format B to Format A's shape
  4. detect plan type: Plan column -> sheet name -> filename suffix
  5. detect patient: form's pt_no -> Pt column -> leading digit of filename
     (falling back to the filename if a single-plan file's Pt column
     disagrees with itself)
  6. coerce numeric columns, standardize ROI, compute Status if absent
  7. drop exact duplicate rows, normalize a "no priority" sentinel to None
     (kept, not dropped — reviewable via engine/corrections.py), compute
     goal_key / structure_class / evaluable

Then, across all files for one case: the HN safeguard, and grouping into
PlanFrames ready for engine.storage.save_case.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import yaml

from engine.config import ROI_ALIASES_PATH
from engine.schemas import (
    CriteriaDirection,
    FormInput,
    GoalRow,
    GoalStatus,
    PlanFrame,
    PlanType,
    StructureClass,
)

__all__ = [
    "ParserError",
    "HNMismatchError",
    "UnrecognizedFileError",
    "ParsedCase",
    "parse_case_files",
    "parse_workbook",
    "resolve_hn",
    "canonical_roi",
    "normalize_roi_text",
    "classify_structure",
    "make_goal_key",
    "compute_status",
    "normalize_plan_token",
    "detect_plan_type",
    "hn_from_filename",
    "patient_no_from_text",
    "NO_PRIORITY_SENTINEL",
]

# RayStation writes Int32.MaxValue when a clinical goal has no priority set.
NO_PRIORITY_SENTINEL = 2147483647

# Header spelling -> canonical column name (matched case/whitespace/
# underscore-insensitively, after stripping header whitespace).
_COLUMN_ALIASES = {
    "pt": "Pt", "patient": "Pt", "patientid": "Pt", "studyid": "Pt", "ptno": "Pt",
    "priority": "Priority",
    "roi": "ROI",
    "goal": "Goal",
    "goaltype": "GoalType",
    "criteria": "Criteria",
    "acceptancelevel": "AcceptanceLevel",
    "parametervalue": "ParameterValue",
    "achievedvalue": "AchievedValue",
    "status": "Status",
    "plan": "Plan", "plantype": "Plan",
}

# The columns a sheet must have (under whichever spelling) to be treated as
# clinical-goal data at all — true for both Format A and (pre-melt) Format B.
_REQUIRED_BASE_COLUMNS = {
    "Priority", "ROI", "Goal", "GoalType", "Criteria",
    "AcceptanceLevel", "ParameterValue",
}

_NON_GOAL_SHEET_NAMES = {"planningtime", "time", "times", "registry", "patients"}

_TARGET_PREFIX_RE = re.compile(r"^(PTV|ITV|CTV|GTV)", re.IGNORECASE)
_NODAL_SUFFIX_RE = re.compile(r"^[\s_-]*n(?![a-zA-Z])", re.IGNORECASE)
_HN_RE = re.compile(r"(?<!\d)(\d{7,9})(?!\d)")
_LEADING_ORDINAL_RE = re.compile(r"^\s*\d+\s+(?=\S)")  # stray "1 " export artifact
_LEADING_Z_RE = re.compile(r"^z(?=[A-Za-z])")  # RayStation "duplicated structure" prefix
_FILENAME_PT_RE = re.compile(r"^\s*(?:pt|patient)?\s*(\d{1,3})(?!\d)", re.IGNORECASE)
_WIDE_ACHIEVED_RE = re.compile(r"^achieved(?:value)?[\s_]*(.+)$", re.IGNORECASE)
_WIDE_STATUS_RE = re.compile(r"^status[\s_]*(.+)$", re.IGNORECASE)
_PLAN_TOKEN_AUTO_MANUAL_RE = re.compile(r"auto\+m|automan|autoandman|auto&man")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ParserError(Exception):
    """Base class for every engine.parser error."""


class HNMismatchError(ParserError):
    """A file's filename HN contradicts the form's HN, files disagree with
    each other, or no HN could be resolved at all."""


class UnrecognizedFileError(ParserError):
    """A sheet has neither a recognizable Format A layout nor a Format B
    (wide, per-plan Achieved_<Plan>) layout."""


class UnreadableFileError(ParserError):
    """The upload isn't a valid Excel workbook at all — wrong file type
    (renamed to .xlsx), corrupted, or password-protected — as opposed to
    UnrecognizedFileError, where the file opens fine but a sheet's layout
    isn't one the parser knows. Raised by _read_workbook so the page layer
    can give a specifically actionable message ("this isn't really an
    Excel file") rather than the generic parse-failure one."""


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #


def hn_from_filename(name: str) -> Optional[str]:
    """'3clinical_goals_54687146_Auto.xlsx' -> '54687146'."""
    m = _HN_RE.search(str(name))
    return m.group(1) if m else None


def patient_no_from_text(text: str) -> Optional[str]:
    """'5clinical_goals_...' or a sheet named 'Pt3' -> 'Pt5' / 'Pt3'."""
    m = _FILENAME_PT_RE.match(str(text))
    return f"Pt{int(m.group(1))}" if m else None


def _format_patient_value(v) -> Optional[str]:
    """A raw Pt-column cell (3, 3.0, '3', 'Pt3', 'PT 3') -> 'Pt3'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    return f"Pt{int(m.group(1))}" if m else s


def normalize_plan_token(text) -> Optional[PlanType]:
    """Tolerant plan-name matcher. Handles 'Auto+Manual', 'AutoManual',
    'Auto+M' (Excel's 31-char sheet-name truncation of '..._Auto+Manual'),
    'auto'/'Auto'/'AUTO', etc. None if nothing recognizable is found."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    t = re.sub(r"[\s_\-]", "", str(text)).lower()
    if _PLAN_TOKEN_AUTO_MANUAL_RE.search(t):
        return PlanType.AUTO_MANUAL
    if "manual" in t:
        return PlanType.MANUAL
    if "auto" in t:
        return PlanType.AUTO
    return None


def normalize_roi_text(raw: str) -> str:
    """Strip export artifacts that are never part of the clinical name: a
    stray leading ordinal ('1 ITV45' -> 'ITV45') and RayStation's 'z' prefix
    for a duplicated/backup structure ('zBone' -> 'Bone')."""
    text = str(raw).strip()
    text = _LEADING_ORDINAL_RE.sub("", text)
    text = _LEADING_Z_RE.sub("", text)
    return text.strip()


@lru_cache(maxsize=1)
def _roi_alias_map() -> dict[str, str]:
    """{lowercased alias spelling: canonical name}, loaded once from
    reference/roi_aliases.yaml."""
    with open(ROI_ALIASES_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    mapping: dict[str, str] = {}
    for canonical, aliases in (data.get("aliases") or {}).items():
        for alias in aliases:
            mapping[str(alias).strip().lower()] = canonical
    return mapping


def canonical_roi(raw: str) -> str:
    """The standardized ROI name used in goal_key / structure_class /
    cross-plan comparisons.

    Per docs/analysis_manual_th_v2.md §2.4's ROI harmonisation table, a
    target (PTV/ITV/CTV/GTV) collapses to just its base type — every dose
    suffix and export quirk dropped ('PTV45'/'PTV 44'/'1 PTV45' -> 'PTV',
    'ITV-T LR'/'zITV' -> 'ITV') — except a nodal boost marker, which keeps
    its own bucket ('PTV-N55' -> 'PTV-N'). A patient is only ever on one
    dose regimen, so "PTV45" and "PTV44" never coexist for the same
    patient; the suffix is regimen-dependent noise, not a second target.
    (Phase 2 originally kept dose suffixes distinct, reasoning they could
    be different SIB targets — corrected once the manual, unavailable at
    the time, confirmed otherwise. See CLAUDE.md rule 1.)

    Non-target structures go through reference/roi_aliases.yaml instead,
    for their own (exact-string) spelling variants.
    """
    normalized = normalize_roi_text(raw)
    target_match = _TARGET_PREFIX_RE.match(normalized)
    if target_match:
        target_type = target_match.group(1).upper()
        remainder = normalized[target_match.end():]
        return f"{target_type}-N" if _NODAL_SUFFIX_RE.match(remainder) else target_type
    return _roi_alias_map().get(normalized.lower(), normalized)


def classify_structure(standardized_roi: str) -> StructureClass:
    """TARGET if the standardized ROI starts with PTV/ITV/CTV/GTV — PTV-N /
    GTV-N included, they are still targets; excluding a nodal boost from a
    primary-coverage *alert* is a Module 4/scoring concern, not a parsing
    one. Everything else is OAR."""
    return StructureClass.TARGET if _TARGET_PREFIX_RE.match(standardized_roi) else StructureClass.OAR


def make_goal_key(roi: str, goal_type: str, criteria: CriteriaDirection,
                   acceptance_level: float, parameter_value: float) -> str:
    """ROI|GoalType|Criteria|AcceptanceLevel|ParameterValue, the last two
    rounded — stable across plans/files for the same clinical goal on the
    same patient (uses the standardized roi, so a spelling difference
    between exports doesn't split one goal into two keys)."""
    criteria_value = criteria.value if isinstance(criteria, CriteriaDirection) else str(criteria)
    return "|".join([
        roi,
        str(goal_type),
        criteria_value,
        f"{round(acceptance_level, 4):.4f}",
        f"{round(parameter_value, 4):.4f}",
    ])


def compute_status(achieved_value: Optional[float], acceptance_level: float,
                    criteria: CriteriaDirection) -> Optional[GoalStatus]:
    """Derive PASS/FAIL from AchievedValue vs AcceptanceLevel and the
    Criteria direction — used when a sheet has no Status column (Format B).
    None (not FAIL) when achieved_value is missing: an unevaluable goal has
    no status yet, it isn't a failure."""
    if achieved_value is None or (isinstance(achieved_value, float) and pd.isna(achieved_value)):
        return None
    if criteria == CriteriaDirection.AT_MOST:
        return GoalStatus.PASS if achieved_value <= acceptance_level + 1e-9 else GoalStatus.FAIL
    return GoalStatus.PASS if achieved_value >= acceptance_level - 1e-9 else GoalStatus.FAIL


# --------------------------------------------------------------------------- #
# Column / layout detection
# --------------------------------------------------------------------------- #


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace, then map known header spellings (case- and
    separator-insensitive) to their canonical name. Unrecognized columns
    are left as-is."""
    df = df.rename(columns=lambda c: str(c).strip())
    rename = {}
    for col in df.columns:
        key = re.sub(r"[\s_]", "", str(col)).lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    return df.rename(columns=rename)


def _is_wide_format(df: pd.DataFrame) -> bool:
    """Format B (combined workbook), wide variant: no single AchievedValue
    column, but one or more Achieved_<Plan> columns instead."""
    if "AchievedValue" in df.columns:
        return False
    for col in df.columns:
        m = _WIDE_ACHIEVED_RE.match(str(col))
        if m and normalize_plan_token(m.group(1)) is not None:
            return True
    return False


def _melt_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Format B (wide) -> Format A's shape: one row per (goal, plan)."""
    achieved_cols: dict[str, PlanType] = {}
    for col in df.columns:
        m = _WIDE_ACHIEVED_RE.match(str(col))
        if not m:
            continue
        plan = normalize_plan_token(m.group(1))
        if plan is not None:
            achieved_cols[col] = plan

    status_col_for_plan: dict[PlanType, str] = {}
    for col in df.columns:
        m = _WIDE_STATUS_RE.match(str(col))
        if m:
            plan = normalize_plan_token(m.group(1))
            if plan is not None:
                status_col_for_plan[plan] = col

    base_cols = [c for c in df.columns
                 if c not in achieved_cols and c not in status_col_for_plan.values()]

    parts = []
    for col, plan in achieved_cols.items():
        part = df[base_cols].copy()
        part["AchievedValue"] = df[col]
        part["Status"] = df[status_col_for_plan[plan]] if plan in status_col_for_plan else np.nan
        part["Plan"] = plan.value
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Plan / patient resolution
# --------------------------------------------------------------------------- #


def _plan_value_or_none(text) -> Optional[str]:
    """normalize_plan_token(text), as a plain str (its .value) rather than
    the PlanType instance — kept out of pandas Series entirely, since a
    str-subclass Enum can get silently flattened to plain str by pandas'
    dtype inference (observed with pandas' pyarrow-backed string dtype),
    which would break a later `.value` lookup."""
    plan = normalize_plan_token(text)
    return plan.value if plan is not None else None


def _resolve_plan(df: pd.DataFrame, *, sheet_name: str, filename: str,
                   warnings: list[str], source: str,
                   plan_type_override: Optional[PlanType] = None) -> pd.DataFrame:
    df = df.copy()
    if plan_type_override is not None:
        # The caller already knows the plan type for certain (e.g. the New
        # Case page's dedicated Manual/Auto/Auto+Manual upload slots) — use
        # it regardless of what the file itself claims, rather than trusting
        # a Plan column, sheet name or filename suffix that could be wrong.
        df["Plan"] = plan_type_override.value
        return df

    if "Plan" in df.columns and df["Plan"].notna().any():
        resolved = df["Plan"].map(_plan_value_or_none)
        fallback = _plan_value_or_none(sheet_name) or _plan_value_or_none(filename)
        if fallback is not None:
            resolved = resolved.where(resolved.notna(), fallback)
        if resolved.isna().any():
            n = int(resolved.isna().sum())
            warnings.append(f"{source}: {n} row(s) have an unrecognized Plan value — dropped")
        df["Plan"] = resolved
        return df[df["Plan"].notna()]

    plan = normalize_plan_token(sheet_name) or normalize_plan_token(filename)
    if plan is None:
        raise UnrecognizedFileError(
            f"{source}: cannot determine plan type from the Plan column, sheet name, "
            "or filename suffix (_Manual/_Auto/_Auto+Manual)"
        )
    df["Plan"] = plan.value
    return df


def _resolve_patient(df: pd.DataFrame, *, sheet_name: str, filename: str,
                      form_pt_no: Optional[str], warnings: list[str], source: str) -> pd.DataFrame:
    df = df.copy()
    if form_pt_no:
        df["Pt"] = form_pt_no
        return df

    filename_pt = patient_no_from_text(filename) or patient_no_from_text(sheet_name)
    single_plan = df["Plan"].nunique(dropna=True) == 1

    if "Pt" in df.columns:
        col = df["Pt"].map(_format_patient_value)
        distinct = sorted(set(col.dropna().unique().tolist()))

        if len(distinct) > 1:
            if not single_plan:
                raise ParserError(
                    f"{source}: Pt column has {len(distinct)} different values and the "
                    "sheet covers more than one plan type — this parser handles one "
                    "patient's case at a time; fix the Pt column or split the file"
                )
            if filename_pt is None:
                raise ParserError(
                    f"{source}: Pt column has {len(distinct)} different values in a "
                    "single-plan file, and no patient number could be read from the "
                    "filename either"
                )
            warnings.append(
                f"{source}: Pt column has {len(distinct)} different values in a "
                f"single-plan file — using '{filename_pt}' from the filename instead"
            )
            df["Pt"] = filename_pt
            return df

        if len(distinct) == 1:
            df["Pt"] = distinct[0]
            return df
        # else: Pt column present but entirely empty -> fall through below

    if filename_pt is None:
        raise ParserError(f"{source}: cannot determine the patient number "
                          "(no usable Pt column, and none in the filename)")
    df["Pt"] = filename_pt
    return df


# --------------------------------------------------------------------------- #
# Per-sheet / per-file parsing
# --------------------------------------------------------------------------- #


def _parse_goal_sheet(df: pd.DataFrame, *, sheet_name: str, filename: str,
                       form_pt_no: Optional[str], warnings: list[str],
                       plan_type_override: Optional[PlanType] = None) -> pd.DataFrame:
    source = f"{filename}/{sheet_name}"
    df = _normalize_columns(df.dropna(how="all"))
    if df.empty:
        return pd.DataFrame()

    base_present = _REQUIRED_BASE_COLUMNS.issubset(df.columns)

    if base_present and "AchievedValue" in df.columns:
        df = _resolve_plan(df, sheet_name=sheet_name, filename=filename, warnings=warnings,
                           source=source, plan_type_override=plan_type_override)
    elif base_present and _is_wide_format(df):
        df = _melt_wide_format(df)
        if df.empty:
            raise UnrecognizedFileError(
                f"{source}: looks like a combined (Format B) sheet but no Achieved_<Plan> "
                "column matched a known plan type"
            )
        warnings.append(f"{source}: combined workbook (Format B) detected — "
                        f"{df['Plan'].nunique()} plan(s)")
        if plan_type_override is not None:
            # A single-plan slot got a multi-plan file — trust what the file
            # itself says (it's the only place that can know all its plans)
            # rather than forcing everything to one type.
            warnings.append(f"{source}: uploaded as {plan_type_override.value}, but this is a "
                            "combined (Format B) file covering more than one plan — using each "
                            "row's own plan type instead")
        df = _resolve_plan(df, sheet_name=sheet_name, filename=filename, warnings=warnings, source=source)
    else:
        raise UnrecognizedFileError(
            f"{source}: not a recognizable clinical-goal sheet "
            f"(missing columns: {sorted(_REQUIRED_BASE_COLUMNS - set(df.columns))})"
        )

    df = _resolve_patient(df, sheet_name=sheet_name, filename=filename,
                          form_pt_no=form_pt_no, warnings=warnings, source=source)

    for col in ("Priority", "AcceptanceLevel", "ParameterValue", "AchievedValue"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Criteria"] = df["Criteria"].astype(str).str.strip()
    df["ROI"] = df["ROI"].astype(str).str.strip()
    df["Goal"] = df["Goal"].astype(str).str.strip()

    if "Status" in df.columns:
        df["Status"] = df["Status"].astype("object")
        df["Status"] = df["Status"].where(df["Status"].notna(), None)
        df["Status"] = df["Status"].map(lambda s: str(s).strip().upper() if s else None)
    else:
        df["Status"] = None

    df["SourceFile"] = source
    return df.reset_index(drop=True)


def _finalize_goals(df: pd.DataFrame, warnings: list[str], *, source_label: str) -> pd.DataFrame:
    """Dedupe, exclude no-priority rows, standardize ROI, compute
    goal_key / structure_class / evaluable, and fall back Status when it
    was missing entirely."""
    if df.empty:
        return df

    before = len(df)
    subset = [c for c in ["Pt", "Plan", "Priority", "ROI", "Goal", "GoalType", "Criteria",
                          "AcceptanceLevel", "ParameterValue", "AchievedValue", "Status"]
              if c in df.columns]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    if len(df) < before:
        warnings.append(f"{source_label}: removed {before - len(df)} exact duplicate row(s)")

    # A missing/sentinel priority is a data-quality problem, not a reason to
    # discard the row: engine/corrections.py needs it in goal_results so it
    # can be reviewed and assigned a real priority (Module 8 / New Case
    # preview). Normalize the sentinel to a genuine missing value (None) and
    # keep the row; scoring/pass-rate exclude it by filtering on Priority
    # being set, not by it having been dropped here.
    no_priority = df["Priority"].isna() | (df["Priority"] == NO_PRIORITY_SENTINEL)
    n_no_priority = int(no_priority.sum())
    if n_no_priority:
        warnings.append(f"{source_label}: {n_no_priority} goal(s) have no protocol priority "
                        "(RayStation sentinel 2147483647) — kept, excluded from scoring until "
                        "a priority is entered (see engine/corrections.py)")
    df["Priority"] = df["Priority"].where(~no_priority, np.nan)

    df["roi_raw"] = df["ROI"]
    df["roi"] = df["ROI"].map(canonical_roi)
    df["structure_class"] = df["roi"].map(classify_structure)

    try:
        df["criteria_enum"] = df["Criteria"].map(CriteriaDirection)
    except ValueError as exc:
        raise ParserError(f"{source_label}: unrecognized Criteria value ({exc})") from exc

    missing_status = df["Status"].isna()
    n_missing_status = int(missing_status.sum())
    if n_missing_status:
        warnings.append(f"{source_label}: {n_missing_status} row(s) had no Status column value "
                        "— computed from AchievedValue vs AcceptanceLevel")
    computed = [
        compute_status(a, acc, c)
        for a, acc, c in zip(df.loc[missing_status, "AchievedValue"],
                             df.loc[missing_status, "AcceptanceLevel"],
                             df.loc[missing_status, "criteria_enum"])
    ]
    df.loc[missing_status, "Status"] = [s.value if s else None for s in computed]

    df["evaluable"] = df["AchievedValue"].notna()
    n_unevaluable = int((~df["evaluable"]).sum())
    if n_unevaluable:
        warnings.append(f"{source_label}: {n_unevaluable} goal(s) have no AchievedValue — kept "
                        "with evaluable=False, flagged for engine/corrections.py")

    df["goal_key"] = [
        make_goal_key(roi, gt, c, acc, pv)
        for roi, gt, c, acc, pv in zip(df["roi"], df["GoalType"], df["criteria_enum"],
                                       df["AcceptanceLevel"], df["ParameterValue"])
    ]
    return df


def _read_workbook(name: str, content: bytes) -> dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(io.BytesIO(content), sheet_name=None)
    except Exception as exc:
        raise UnreadableFileError(f"{name}: cannot read as an Excel workbook ({exc})") from exc


def detect_plan_type(name: str, content: bytes) -> Optional[PlanType]:
    """Best-effort plan-type detection for one uploaded file, for UI use
    before the user has committed it to a Manual/Auto/Auto+Manual slot
    (pages/0_new_case.py's multi-file uploader) — NOT used during real
    parsing, which instead trusts an explicit plan_type_override once a
    type has been decided (by this detection, or by the user resolving a
    conflict/unknown in the UI).

    Applies the same cascade _parse_goal_sheet/_resolve_plan apply per
    sheet during real parsing — a Plan column's value, then the sheet
    name, then the filename — but at the whole-file level: every
    goal-bearing sheet must agree (or supply no signal at all) for a
    result to come back.

    Returns None — "couldn't determine it, don't guess" — for an
    unreadable file, a file with no goal-bearing sheet giving any signal,
    or sheets whose Plan column/sheet name disagree with each other.
    Never raises: the real parse_case_files() call (once the user has
    confirmed a type for every file) is what raises for a genuinely bad
    upload; this is only ever a hint.
    """
    try:
        book = _read_workbook(name, content)
    except ParserError:
        return None

    candidates: set[PlanType] = set()
    for sheet_name, sheet_df in book.items():
        if re.sub(r"[\s_]", "", str(sheet_name)).lower() in _NON_GOAL_SHEET_NAMES:
            continue

        plan = None
        df = _normalize_columns(sheet_df)
        if "Plan" in df.columns and df["Plan"].notna().any():
            resolved = {normalize_plan_token(v) for v in df["Plan"].dropna().unique()}
            resolved.discard(None)
            if len(resolved) == 1:
                plan = next(iter(resolved))
        if plan is None:
            plan = normalize_plan_token(sheet_name)
        if plan is not None:
            candidates.add(plan)

    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        return None  # sheets disagree with each other -- don't guess

    return normalize_plan_token(name)


def parse_workbook(name: str, content: bytes, *, form_pt_no: Optional[str],
                    warnings: list[str], plan_type_override: Optional[PlanType] = None) -> pd.DataFrame:
    """Parse every goal sheet in one uploaded workbook into a single long
    DataFrame (post-dedupe, with goal_key/structure_class/evaluable). A
    sheet whose layout isn't recognizable as clinical-goal data is skipped
    with a warning (e.g. an incidental notes sheet); any other problem
    (ambiguous patient, unresolved plan type, bad Criteria value) raises.

    plan_type_override: pass this when the caller already knows the plan
    type for certain — e.g. the New Case page's dedicated Manual/Auto/
    Auto+Manual upload slots — so a Format-A sheet's plan type is taken
    from the slot rather than guessed from its Plan column/sheet name/
    filename. Ignored for a Format B (combined, multi-plan) sheet.
    """
    book = _read_workbook(name, content)
    blocks = []
    for sheet_name, sheet_df in book.items():
        if re.sub(r"[\s_]", "", str(sheet_name)).lower() in _NON_GOAL_SHEET_NAMES:
            continue
        try:
            parsed = _parse_goal_sheet(sheet_df, sheet_name=sheet_name, filename=name,
                                       form_pt_no=form_pt_no, warnings=warnings,
                                       plan_type_override=plan_type_override)
        except UnrecognizedFileError as exc:
            warnings.append(str(exc))
            continue
        if not parsed.empty:
            blocks.append(parsed)

    if not blocks:
        raise ParserError(f"{name}: no clinical-goal data found in any sheet")

    combined = pd.concat(blocks, ignore_index=True)
    return _finalize_goals(combined, warnings, source_label=name)


# --------------------------------------------------------------------------- #
# HN safeguard
# --------------------------------------------------------------------------- #


def resolve_hn(filenames: Sequence[str], form_hn: Optional[str]) -> tuple[str, list[str]]:
    """Extract a 7-9 digit HN from each filename and reconcile it against
    the form. Raises HNMismatchError if the form's HN contradicts a
    filename's, or if HN wasn't entered and either the files disagree with
    each other or none of them carries one. Returns (resolved_hn, warnings).
    """
    warnings: list[str] = []
    found = {name: hn for name in filenames if (hn := hn_from_filename(name))}

    if form_hn:
        mismatched = {name: hn for name, hn in found.items() if hn != form_hn}
        if mismatched:
            raise HNMismatchError(
                f"HN on the form ({form_hn}) does not match the HN in filename(s): "
                + ", ".join(f"{n} ({hn})" for n, hn in mismatched.items())
            )
        return form_hn, warnings

    distinct = set(found.values())
    if len(distinct) == 1:
        resolved = next(iter(distinct))
        warnings.append(f"HN not entered on the form — using {resolved}, found in filename(s)")
        return resolved, warnings
    if len(distinct) > 1:
        raise HNMismatchError(
            "HN not entered on the form, and the uploaded files disagree on it: "
            + ", ".join(f"{n} ({hn})" for n, hn in found.items())
        )
    raise HNMismatchError(
        "HN not entered on the form, and no 7-9 digit HN could be found in any filename"
    )


# --------------------------------------------------------------------------- #
# Case-level orchestration
# --------------------------------------------------------------------------- #


@dataclass
class ParsedCase:
    """Ready-to-persist output of parsing one case's uploaded files."""
    plan_frames: list[PlanFrame]
    resolved_hn: str
    warnings: list[str] = field(default_factory=list)


# Accepted as one uploaded file: a filesystem path, a (name, bytes) pair, or
# anything Streamlit-UploadedFile-shaped (.name / .getvalue()).
FileInput = Union[str, Path, tuple]


def _read_file_input(f: FileInput) -> tuple[str, bytes]:
    if hasattr(f, "name") and hasattr(f, "getvalue"):
        return f.name, f.getvalue()
    if isinstance(f, (str, Path)):
        p = Path(f)
        return p.name, p.read_bytes()
    name, content = f
    return name, content


def parse_case_files(files: Sequence[FileInput], form: FormInput, *,
                     plan_type_overrides: Optional[Sequence[Optional[PlanType]]] = None) -> ParsedCase:
    """Parse every uploaded file for one case (Module 0 — New Case) into
    PlanFrames ready for engine.storage.save_case, applying the HN
    safeguard across all of them.

    plan_type_overrides, when given, must be the same length as `files` —
    each entry is the known plan type for that file (or None to fall back
    to auto-detection), for a caller with dedicated per-plan upload slots.
    See parse_workbook.
    """
    if not files:
        raise ParserError("no files given")
    if plan_type_overrides is not None and len(plan_type_overrides) != len(files):
        raise ParserError("plan_type_overrides must be the same length as files")

    read = [_read_file_input(f) for f in files]
    resolved_hn, warnings = resolve_hn([name for name, _ in read], form.hn)

    overrides = plan_type_overrides or [None] * len(read)
    per_file = []
    for (name, content), override in zip(read, overrides):
        df = parse_workbook(name, content, form_pt_no=form.pt_no, warnings=warnings,
                            plan_type_override=override)
        sha256 = hashlib.sha256(content).hexdigest()
        per_file.append((name, sha256, df))

    plan_frames = []
    for plan_type in PlanType:
        contributing = [(name, sha, df[df["Plan"] == plan_type.value])
                        for name, sha, df in per_file if (df["Plan"] == plan_type.value).any()]
        if not contributing:
            continue

        plan_df = pd.concat([d for _, _, d in contributing], ignore_index=True)
        source_name = contributing[0][0] if len(contributing) == 1 else \
            "; ".join(n for n, _, _ in contributing)
        sha = contributing[0][1] if len(contributing) == 1 else None

        goals = [
            GoalRow(
                goal_key=row.goal_key,
                priority=None if pd.isna(row.Priority) else int(row.Priority),
                roi_raw=row.roi_raw,
                roi=row.roi,
                goal_text=row.Goal,
                goal_type=row.GoalType,
                criteria=row.criteria_enum,
                acceptance_level=float(row.AcceptanceLevel),
                parameter_value=float(row.ParameterValue),
                achieved_value=None if pd.isna(row.AchievedValue) else float(row.AchievedValue),
                status=GoalStatus(row.Status) if row.Status else None,
                evaluable=bool(row.evaluable),
                structure_class=row.structure_class,
            )
            for row in plan_df.itertuples()
        ]
        plan_frames.append(PlanFrame(
            plan_type=plan_type,
            planning_time_min=form.time_for(plan_type),
            is_straight_pass=None,
            source_filename=source_name,
            file_sha256=sha,
            goals=goals,
        ))

    return ParsedCase(plan_frames=plan_frames, resolved_hn=resolved_hn, warnings=warnings)
