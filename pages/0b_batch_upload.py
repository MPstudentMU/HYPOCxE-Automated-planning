"""Batch Upload — Module 0b (docs/analysis_manual_th_v2.md §2.2-2.3).

Upload every plan file for however many patients at once — no per-patient
form up front (that's what Module 1's Edit dialog is for, once a case
exists). Each file's patient number, HN, and plan type are detected with
engine.parser's existing cascade (never duplicated — engine.batch just
composes it); files needing a decision get one, then engine.batch.
group_batch_files classifies each patient's files as New, Duplicate (an
existing pt_no — Skip or Overwrite, your call), or Needs review (per
§2.3, a patient's own files disagreeing on HN — blocked alone, without
holding up the rest of the batch). One "Confirm batch save" commits
every row that isn't blocked. Display + orchestration only: all detection,
grouping, and classification is engine.batch/engine.parser; all
persistence is engine.storage.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from components.pending_preview import apply_pending_corrections, render_pending_preview
from engine import parser as P
from engine.batch import BatchFile, BatchGroup, detect_batch_files, group_batch_files
from engine.schemas import FormInput, PlanType
from engine.storage import replace_plan, save_case

st.title("Batch Upload")
st.caption("Module 0b — upload multiple patients' plan files at once")

engine = get_engine()
render_pending_banner(engine)

st.caption(
    "Drop plan files for as many patients as you like in one go. Each file's patient "
    "number (leading digit of the filename), HN (7-9 digit number in the filename), "
    "and plan type are detected automatically — the same detection New Case uses. "
    "Case details beyond HN and the plan files themselves (dose regimen, tx room, "
    "MP1/MP2/RO, planning time) aren't collected here; fill those in afterward via "
    "Patient Data's Edit dialog."
)

CHOOSE = "— choose —"
SKIP = "Skip"
PLAN_LABELS = [p.value for p in PlanType]


@st.cache_data(show_spinner=False)
def _detect_all(files: list[tuple[str, bytes]]) -> list[BatchFile]:
    # Cached on the exact set of (name, content) uploaded: without this,
    # every rerun (any widget interaction anywhere on the page) would
    # re-open and re-scan every workbook in the batch again.
    return detect_batch_files(files)


uploaded_files = st.file_uploader(
    "Plan files (multiple patients)", type=["xlsx"], accept_multiple_files=True,
    key="batch_files",
)
if not uploaded_files:
    st.stop()

raw_files = [(f.name, f.getvalue()) for f in uploaded_files]
detected = _detect_all(raw_files)
pairs = list(zip(uploaded_files, detected))  # (UploadedFile, BatchFile), same order

# --------------------------------------------------------------------------- #
# Phase 1 — resolve any file whose patient number couldn't be detected
# --------------------------------------------------------------------------- #

unresolved_pt_pairs = [(uf, bf) for uf, bf in pairs if bf.pt_no is None]
if unresolved_pt_pairs:
    n = len(unresolved_pt_pairs)
    st.warning(
        f"⚠️ Couldn't determine a patient number for {n} file{'s' if n != 1 else ''} — "
        "assign one below (leave blank to exclude it from this batch).",
        icon="⚠️",
    )
    for uf, bf in unresolved_pt_pairs:
        manual = st.text_input(
            f"Patient no. for **{uf.name}**", placeholder="Pt7",
            key=f"batch_ptno_{uf.file_id}",
        )
        bf.pt_no = manual.strip() or None

still_unassigned = [uf for uf, bf in pairs if bf.pt_no is None]
if still_unassigned:
    st.caption(
        f"{len(still_unassigned)} file(s) still have no patient number and are excluded "
        "from this batch until one is provided above."
    )

# --------------------------------------------------------------------------- #
# Phase 2 — group by (resolved) patient number, resolve plan-type conflicts
# within each group (the same selectbox flow pages/0_new_case.py uses,
# scoped per patient instead of globally)
# --------------------------------------------------------------------------- #

raw_groups: dict[str, list[tuple]] = defaultdict(list)
for uf, bf in pairs:
    if bf.pt_no is not None:
        raw_groups[bf.pt_no].append((uf, bf))


def _resolve_group_plan_types(pt_no: str, group_pairs: list[tuple]) -> tuple[dict[str, PlanType], bool]:
    """Returns ({file_id: PlanType}, fully_resolved). Renders a selectbox
    for any file whose plan type is ambiguous/undetected within this one
    patient's files — mirrors pages/0_new_case.py's own resolution UI."""
    by_type: dict[PlanType, list] = defaultdict(list)
    for uf, bf in group_pairs:
        if bf.plan_type is not None:
            by_type[bf.plan_type].append(uf)
    conflicting_ids = {uf.file_id for group in by_type.values() if len(group) > 1 for uf in group}
    undetected_ids = {uf.file_id for uf, bf in group_pairs if bf.plan_type is None}
    needs_selectbox_ids = conflicting_ids | undetected_ids

    resolved: dict[str, PlanType] = {}
    for uf, bf in group_pairs:
        if uf.file_id not in needs_selectbox_ids:
            resolved[uf.file_id] = bf.plan_type

    incomplete = False
    if needs_selectbox_ids:
        n = len(needs_selectbox_ids)
        with st.expander(f"⚠️ {pt_no}: assign a plan type for {n} file(s)", expanded=True):
            for uf, bf in group_pairs:
                if uf.file_id not in needs_selectbox_ids:
                    continue
                guess = bf.plan_type
                reason = ("detected the same plan type as another file for this patient"
                         if uf.file_id in conflicting_ids
                         else "couldn't detect a plan type")
                row1, row2 = st.columns([3, 1])
                guess_note = f" — guessed **{guess.value}**" if guess is not None else ""
                row1.markdown(f"**{uf.name}** — {reason}{guess_note}")
                options = [CHOOSE, *PLAN_LABELS, SKIP]
                default_index = options.index(guess.value) if guess is not None else 0
                choice = row2.selectbox(
                    "Assign to", options, index=default_index,
                    key=f"batch_assign_{uf.file_id}", label_visibility="collapsed",
                )
                if choice == CHOOSE:
                    incomplete = True
                elif choice != SKIP:
                    resolved[uf.file_id] = PlanType(choice)

    final_by_type: dict[PlanType, list] = defaultdict(list)
    for fid, pt in resolved.items():
        final_by_type[pt].append(fid)
    has_duplicates = any(len(v) > 1 for v in final_by_type.values())
    return resolved, (not incomplete and not has_duplicates)


resolved_files: list[BatchFile] = []
plan_type_ready: dict[str, bool] = {}
for pt_no, group_pairs in raw_groups.items():
    resolved_types, ready = _resolve_group_plan_types(pt_no, group_pairs)
    plan_type_ready[pt_no] = ready
    for uf, bf in group_pairs:
        if uf.file_id in resolved_types:
            resolved_files.append(BatchFile(
                name=bf.name, content=bf.content, pt_no=pt_no, hn=bf.hn,
                plan_type=resolved_types[uf.file_id],
            ))

# --------------------------------------------------------------------------- #
# Phase 3 — classify each ready patient (New / Duplicate / Needs review);
# a patient still blocked on plan-type assignment is its own Needs review
# row rather than silently dropped from the summary
# --------------------------------------------------------------------------- #

ready_files = [f for f in resolved_files if plan_type_ready[f.pt_no]]
groups = group_batch_files(engine, ready_files) if ready_files else []

for pt_no, group_pairs in raw_groups.items():
    if not plan_type_ready[pt_no]:
        groups.append(BatchGroup(
            pt_no=pt_no,
            files=[BatchFile(name=bf.name, content=bf.content, pt_no=pt_no, hn=bf.hn,
                             plan_type=bf.plan_type) for _uf, bf in group_pairs],
            status="needs_review", reason="Plan type not yet fully assigned above",
        ))

groups.sort(key=lambda g: g.pt_no)

# --------------------------------------------------------------------------- #
# Phase 4 — Duplicate: Skip/Overwrite per row
# --------------------------------------------------------------------------- #

duplicate_choice: dict[str, str] = {}
duplicate_groups = [g for g in groups if g.status == "duplicate"]
if duplicate_groups:
    st.subheader("Existing patients found")
    for g in duplicate_groups:
        choice = st.radio(
            f"**{g.pt_no}** already exists — what should this batch do with its files?",
            ["Skip", "Overwrite"], horizontal=True, key=f"batch_dup_{g.pt_no}",
        )
        duplicate_choice[g.pt_no] = choice


def _is_actionable(g) -> bool:
    if g.status == "new":
        return True
    if g.status == "duplicate":
        return duplicate_choice.get(g.pt_no) == "Overwrite"
    return False


# --------------------------------------------------------------------------- #
# Phase 5 — parse every actionable group now (so the summary table can show
# a real pending-correction count), catching a genuine parse failure per
# group rather than letting it take down the whole batch
# --------------------------------------------------------------------------- #

parsed_by_pt_no: dict[str, P.ParsedCase] = {}
for g in groups:
    if not _is_actionable(g):
        continue
    files = [(f.name, f.content) for f in g.files]
    overrides = [f.plan_type for f in g.files]
    form = FormInput(pt_no=g.pt_no, hn=g.resolved_hn)
    try:
        parsed_by_pt_no[g.pt_no] = P.parse_case_files(files, form, plan_type_overrides=overrides)
    except P.ParserError as exc:
        g.status, g.reason = "needs_review", f"Could not parse: {exc}"

# --------------------------------------------------------------------------- #
# Phase 6 — one summary table, before anything is saved
# --------------------------------------------------------------------------- #

st.subheader("Summary")
if not groups:
    st.info("Upload files above to see patients detected here.")
    st.stop()

STATUS_LABELS = {
    "new": "🆕 New",
    "duplicate": "🔁 Duplicate",
    "needs_review": "⚠️ Needs review",
}
rows = []
for g in groups:
    plan_types_present = sorted({f.plan_type.value for f in g.files if f.plan_type is not None},
                                key=PLAN_LABELS.index)
    status_label = STATUS_LABELS[g.status]
    if g.status == "duplicate":
        status_label += f" — {duplicate_choice.get(g.pt_no, 'Skip')}"
    elif g.status == "needs_review":
        status_label += f" ({g.reason})"

    parsed = parsed_by_pt_no.get(g.pt_no)
    if parsed is not None:
        pending_n = sum(1 for frame in parsed.plan_frames for goal in frame.goals
                        if goal.achieved_value is None or goal.priority is None)
        pending = str(pending_n)
    else:
        pending = "—"  # not parsed (blocked, or a skipped duplicate) -- not "0", a real unknown

    rows.append(dict(
        Patient=g.pt_no,
        HN=g.resolved_hn or "—",
        **{"Detected plans": ", ".join(plan_types_present) if plan_types_present else "—"},
        Status=status_label,
        **{"Pending corrections": pending},
    ))

st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Phase 7 — per-patient correction editor, only where there's something
# pending (reuses pages/0_new_case.py's own pre-save editor)
# --------------------------------------------------------------------------- #

edited_pending_by_pt_no: dict[str, pd.DataFrame] = {}
actionable_with_pending = [
    g for g in groups if _is_actionable(g) and g.pt_no in parsed_by_pt_no
    and any(goal.achieved_value is None or goal.priority is None
           for frame in parsed_by_pt_no[g.pt_no].plan_frames for goal in frame.goals)
]
if actionable_with_pending:
    st.subheader("Goals needing review")
    for g in actionable_with_pending:
        with st.expander(f"{g.pt_no}"):
            for w in parsed_by_pt_no[g.pt_no].warnings:
                st.info(w, icon="ℹ️")
            edited_pending_by_pt_no[g.pt_no] = render_pending_preview(
                parsed_by_pt_no[g.pt_no].plan_frames, key=f"batch_{g.pt_no}",
            )

# --------------------------------------------------------------------------- #
# Phase 8 — confirm and save every actionable row
# --------------------------------------------------------------------------- #

any_filled = any(
    not edited.empty and edited.apply(
        lambda r: pd.notna(r["value"]) or bool(r["confirmed_not_evaluable"]) or pd.notna(r["priority"]),
        axis=1,
    ).any()
    for edited in edited_pending_by_pt_no.values()
)

corrected_by = ""
if any_filled:
    corrected_by = st.text_input(
        "Your name (required to save any correction filled in above)",
        value=st.session_state.get("entered_by", ""), key="batch_corrected_by",
    )

actionable_groups = [g for g in groups if _is_actionable(g)]
st.divider()
st.caption(f"{len(actionable_groups)} patient(s) will be saved; "
          f"{sum(1 for g in groups if g.status == 'needs_review')} blocked (Needs review); "
          f"{sum(1 for g in groups if g.status == 'duplicate' and not _is_actionable(g))} skipped (Duplicate).")

if st.button("Confirm batch save", type="primary", disabled=not actionable_groups):
    if any_filled and not corrected_by.strip():
        st.error("Enter your name before saving — at least one correction was filled in.")
    else:
        entered_by = st.session_state.get("entered_by")
        saved_patients, total_corrections, errors = [], 0, []
        for g in actionable_groups:
            parsed = parsed_by_pt_no.get(g.pt_no)
            if parsed is None:
                continue  # this group's parse failed and was downgraded to needs_review above
            try:
                if g.status == "duplicate":
                    for frame in parsed.plan_frames:
                        replace_plan(engine, g.existing_patient_id, frame)
                else:
                    save_case(engine, FormInput(pt_no=g.pt_no, hn=g.resolved_hn),
                             parsed.plan_frames, entered_by=entered_by)

                edited = edited_pending_by_pt_no.get(g.pt_no, pd.DataFrame())
                saved, warns, errs = apply_pending_corrections(
                    engine, g.pt_no, edited, corrected_by=corrected_by,
                )
                total_corrections += saved
                errors.extend(warns)
                errors.extend(errs)
                saved_patients.append(g.pt_no)
            except Exception as exc:  # noqa: BLE001 — surface it, don't stop the rest of the batch
                errors.append(f"{g.pt_no}: could not save this patient ({exc})")

        for e in errors:
            st.warning(e)
        if saved_patients:
            st.success(
                f"Saved {len(saved_patients)} patient(s): {', '.join(saved_patients)}"
                f"{f' ({total_corrections} correction(s))' if total_corrections else ''}."
            )
            st.caption("Upload another batch above whenever you're ready.")
