"""Batch upload grouping/classification — pages/0b_batch_upload.py's
non-Streamlit logic (§2.2-2.3 of docs/analysis_manual_th_v2.md), kept here
so it's testable without a running app and doesn't duplicate anything
engine/parser.py already does.

detect_batch_files() runs the *existing* per-file detection cascade —
engine.parser.patient_no_from_text (leading digit of the filename),
hn_from_filename (7-9 digit number in the filename), detect_plan_type
(Plan column -> sheet name -> filename) — over every uploaded file. It
never guesses beyond what those already do; a file left with pt_no or
plan_type still None is exactly the signal pages/0b_batch_upload.py uses
to ask the user, the same way pages/0_new_case.py already does for plan
type alone.

group_batch_files() takes files whose pt_no and plan_type are already
resolved (by detection, or by the page's own selectbox/text_input
fallback — this function never resolves anything itself) and classifies
each patient's group as "new", "duplicate" (pt_no already in the
database — Skip/Overwrite is the caller's decision), or "needs_review":
per the manual's §2.3 table — "นำเข้าแบบกลุ่ม (Batch) แล้วพบผู้ป่วยหนึ่งรายมีหลาย HN
-> บันทึกคำเตือนให้ผู้วิจัยตรวจสอบ" ("batch import, one patient found with several
HNs -> log a warning for the researcher to review") — a group whose files
disagree on HN is flagged, not guessed at, and blocks only that one
patient's row, never the rest of the batch. The HN check itself is just
engine.parser.resolve_hn (also unduplicated) called per group with no
form_hn, so it raises exactly when the files disagree or none carries one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.engine import Engine

from engine import parser as P
from engine.schemas import PlanType
from engine.storage import find_patient_by_pt_no

__all__ = ["BatchFile", "BatchGroup", "detect_batch_files", "group_batch_files"]


@dataclass
class BatchFile:
    """One uploaded file's detected (or user-assigned) identity."""

    name: str
    content: bytes
    pt_no: Optional[str]
    hn: Optional[str]
    plan_type: Optional[PlanType]


@dataclass
class BatchGroup:
    """One patient's worth of BatchFiles, classified for
    pages/0b_batch_upload.py's summary table."""

    pt_no: str
    files: list[BatchFile]
    status: str  # "new" | "duplicate" | "needs_review"
    reason: Optional[str] = None  # set for "needs_review"
    resolved_hn: Optional[str] = None  # set for "new" / "duplicate"
    existing_patient_id: Optional[int] = None  # set for "duplicate"


def detect_batch_files(files: list[tuple[str, bytes]]) -> list[BatchFile]:
    """files: (filename, content) pairs, in whatever order they were
    uploaded — order never matters here, only the filename/content of
    each. Detection only; doesn't group or classify anything."""
    return [
        BatchFile(
            name=name, content=content,
            pt_no=P.patient_no_from_text(name),
            hn=P.hn_from_filename(name),
            plan_type=P.detect_plan_type(name, content),
        )
        for name, content in files
    ]


def group_batch_files(engine: Engine, files: list[BatchFile]) -> list[BatchGroup]:
    """Group already-resolved BatchFiles (pt_no and plan_type both set —
    see module docstring) by pt_no, and classify each group.

    A group whose files still share a plan_type is also flagged
    "needs_review" (with its own reason) as a defensive fallback — the
    intended path is that pages/0b_batch_upload.py resolves plan-type
    conflicts (the same selectbox flow pages/0_new_case.py already uses)
    before a group ever reaches here, so this should be unreachable in
    practice, but grouping must never silently pick one file over another.
    """
    by_pt_no: dict[str, list[BatchFile]] = defaultdict(list)
    for f in files:
        by_pt_no[f.pt_no].append(f)

    groups = []
    for pt_no, group_files in by_pt_no.items():
        plan_types = [f.plan_type for f in group_files]
        if len(plan_types) != len(set(plan_types)):
            groups.append(BatchGroup(
                pt_no=pt_no, files=group_files, status="needs_review",
                reason="More than one file detected as the same plan type",
            ))
            continue

        try:
            resolved_hn, _warnings = P.resolve_hn([f.name for f in group_files], None)
        except P.HNMismatchError as exc:
            groups.append(BatchGroup(
                pt_no=pt_no, files=group_files, status="needs_review", reason=str(exc),
            ))
            continue

        existing = find_patient_by_pt_no(engine, pt_no)
        if existing is not None:
            groups.append(BatchGroup(
                pt_no=pt_no, files=group_files, status="duplicate",
                resolved_hn=resolved_hn, existing_patient_id=existing.id,
            ))
        else:
            groups.append(BatchGroup(
                pt_no=pt_no, files=group_files, status="new", resolved_hn=resolved_hn,
            ))

    return groups
