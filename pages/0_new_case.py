"""New Case — Module 0.

Intake: fill the case form, upload the plan export file(s), preview what
was parsed (goal counts, warnings, anything needing a correction), then
confirm to save. All parsing, validation and persistence lives in engine/;
this file only renders the form, the preview, and the editor.

Plan export files are uploaded as one batch (rather than three dedicated
Manual/Auto/Auto+Manual slots) and each file's plan type is detected with
engine.parser.detect_plan_type() — the same Plan column -> sheet name ->
filename cascade real parsing uses (engine.parser._resolve_plan), just
applied once per file instead of once per sheet. Detection only ever
produces a *default*: if two files land on the same type, or a file can't
be placed at all, the page shows a selectbox for it rather than guessing,
and Preview stays disabled until every file has a definite plan type (or
is explicitly skipped) with no two files sharing one.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from components.pending_preview import apply_pending_corrections, render_pending_preview
from engine import parser as P
from engine.config import DOSE_REGIMEN_UNSET, dose_regimen_choices, rx_for_dose_regimen
from engine.schemas import FormInput, PlanType
from engine.storage import find_patient_by_pt_no, replace_plan, save_case, update_patient

st.title("New Case")
st.caption("Module 0 — intake of a new planning case")

engine = get_engine()
render_pending_banner(engine)

# Only pt_no and the plan files are actually required to save a case —
# everything else here (dose regimen, tx room, MP1/MP2/RO, planning time)
# can be left blank and filled in later via Module 1's Edit dialog. See
# engine.schemas.FormInput and engine.export.load_registry_frame's
# profile_complete.
DOSE_CHOICES = dose_regimen_choices()

CHOOSE = "— choose —"
SKIP = "Skip"
PLAN_LABELS = [p.value for p in PlanType]
PLAN_ORDER = [PlanType.MANUAL, PlanType.AUTO, PlanType.AUTO_MANUAL]


@st.cache_data(show_spinner=False)
def _detect(name: str, content: bytes):
    # Cached per (name, content): without this, every rerun (any widget
    # interaction anywhere on the page, not just the uploader) would
    # re-open and re-scan every uploaded workbook again.
    return P.detect_plan_type(name, content)


st.subheader("Case details")
c1, c2 = st.columns(2)
with c1:
    hn = st.text_input("HN", help="Leave blank to use the value from the uploaded filenames.")
    pt_no = st.text_input("Patient no.", placeholder="Pt7")
    dose_choice = st.radio(
        "Dose regimen", [DOSE_REGIMEN_UNSET, *DOSE_CHOICES.keys()],
        help="Optional at intake — leave unset and fill it in later via Edit (Module 1) if "
            "you don't have it yet.",
    )
    sib_boost = st.checkbox("SIB boost")
    overwrite = st.checkbox(
        "Overwrite if this patient no. already exists",
        help="Without this, saving a case with a patient no. that's already in the "
            "database is rejected — this must be checked to intentionally replace it.",
    )
with c2:
    tx_room = st.text_input("Treatment room")
    mp1 = st.text_input("Medical physicist 1")
    mp2 = st.text_input("Medical physicist 2")
    ro = st.text_input("Radiation oncologist")

st.subheader("Planning time (minutes)")
t1, t2, t3 = st.columns(3)
time_manual = t1.number_input("Manual", min_value=0.0, step=1.0, value=0.0)
time_auto = t2.number_input("Auto", min_value=0.0, step=1.0, value=0.0)
time_automanual = t3.number_input(
    "Auto+Manual", min_value=0.0, step=1.0, value=0.0,
    help="The *total* planning time for the Auto+Manual plan, including the Auto step — "
        "not just the additional manual touch-up time on top of it.",
)

st.subheader("Plan export files")
st.caption(
    "RayStation clinical-goal export — upload every plan file for this case at once "
    "(Manual, Auto, and optionally Auto+Manual). Each file's plan type is detected "
    "automatically from its Plan column, sheet name, or filename suffix; you'll be "
    "asked to assign anything it can't confidently place."
)
uploaded_files = st.file_uploader(
    "Plan files", type=["xlsx"], accept_multiple_files=True,
    key="new_case_files", label_visibility="collapsed",
)

# --------------------------------------------------------------------------- #
# Detect each file's plan type; resolve conflicts/unknowns via a selectbox
# --------------------------------------------------------------------------- #

assignment: dict[str, PlanType] = {}   # file_id -> resolved plan type (Skip excluded)
file_by_id = {f.file_id: f for f in uploaded_files} if uploaded_files else {}
selection_incomplete = False
duplicates: dict[PlanType, list[str]] = {}

if uploaded_files:
    detected = {f.file_id: _detect(f.name, f.getvalue()) for f in uploaded_files}

    by_type: dict[PlanType, list] = defaultdict(list)
    for f in uploaded_files:
        if detected[f.file_id] is not None:
            by_type[detected[f.file_id]].append(f)
    conflicting_ids = {f.file_id for group in by_type.values() if len(group) > 1 for f in group}
    undetected_ids = {f.file_id for f in uploaded_files if detected[f.file_id] is None}
    needs_selectbox_ids = conflicting_ids | undetected_ids

    for f in uploaded_files:
        if f.file_id not in needs_selectbox_ids:
            assignment[f.file_id] = detected[f.file_id]

    if needs_selectbox_ids:
        n = len(needs_selectbox_ids)
        st.warning(
            f"⚠️ Couldn't confidently place {n} file{'s' if n != 1 else ''} — assign "
            "each one below.",
            icon="⚠️",
        )
        for f in uploaded_files:
            if f.file_id not in needs_selectbox_ids:
                continue
            guess = detected[f.file_id]
            reason = ("detected the same plan type as another file below"
                     if f.file_id in conflicting_ids
                     else "couldn't detect a plan type (no Plan column, sheet name, or "
                          "filename match)")
            row1, row2 = st.columns([3, 1])
            guess_note = f" — guessed **{guess.value}**" if guess is not None else ""
            row1.markdown(f"**{f.name}** — {reason}{guess_note}")
            options = [CHOOSE, *PLAN_LABELS, SKIP]
            default_index = options.index(guess.value) if guess is not None else 0
            choice = row2.selectbox(
                "Assign to", options, index=default_index,
                key=f"new_case_assign_{f.file_id}", label_visibility="collapsed",
            )
            if choice == CHOOSE:
                selection_incomplete = True
            elif choice != SKIP:
                assignment[f.file_id] = PlanType(choice)

    final_by_type: dict[PlanType, list[str]] = defaultdict(list)
    for file_id, plan_type in assignment.items():
        final_by_type[plan_type].append(file_by_id[file_id].name)
    duplicates = {pt: names for pt, names in final_by_type.items() if len(names) > 1}

    if duplicates:
        st.error(
            "⚠️ More than one file is still assigned to the same plan type — fix the "
            "assignment(s) above: "
            + "; ".join(f"{pt.value}: {', '.join(names)}" for pt, names in duplicates.items())
        )
    elif not selection_incomplete:
        summary = ", ".join(
            f"{pt.value} ({file_by_id[fid].name})"
            for pt in PLAN_ORDER
            for fid, assigned_pt in assignment.items() if assigned_pt == pt
        )
        st.caption(f"Detected: {summary}" if summary else "No plan files recognized yet.")

have_manual = any(pt == PlanType.MANUAL for pt in assignment.values())
have_auto = any(pt == PlanType.AUTO for pt in assignment.values())
if uploaded_files and not selection_incomplete and not duplicates and not (have_manual and have_auto):
    st.caption("Both a Manual and an Auto plan file are required (Auto+Manual is optional).")

is_ready = bool(uploaded_files) and not selection_incomplete and not duplicates and have_manual and have_auto
preview_clicked = st.button("Preview", disabled=not is_ready, type="primary")

if preview_clicked:
    if not pt_no.strip():
        st.error("Enter a patient number.")
    elif not (have_manual and have_auto):
        st.error("Both the Manual and Auto plan files are required (Auto+Manual is optional).")
    else:
        existing = find_patient_by_pt_no(engine, pt_no.strip())
        if existing and not overwrite:
            st.error(
                f"Patient no. '{pt_no.strip()}' already exists. Check "
                "'Overwrite if this patient no. already exists' to replace it."
            )
        else:
            dose_regimen = DOSE_CHOICES.get(dose_choice)  # None if left unset
            rx_cgy, fractions = rx_for_dose_regimen(dose_regimen) if dose_regimen else (None, None)
            try:
                form = FormInput(
                    hn=hn.strip() or None,
                    pt_no=pt_no.strip(),
                    dose_regimen=dose_regimen,
                    rx_cgy=rx_cgy,
                    fractions=fractions,
                    sib_boost=sib_boost,
                    tx_room=tx_room.strip() or None,
                    mp1=mp1.strip() or None,
                    mp2=mp2.strip() or None,
                    ro=ro.strip() or None,
                    time_manual=time_manual or None,
                    time_auto=time_auto or None,
                    time_automanual=time_automanual or None,
                )
                files = []
                overrides = []
                for plan_type in PLAN_ORDER:
                    file_id = next((fid for fid, pt in assignment.items() if pt == plan_type), None)
                    if file_id is not None:
                        f = file_by_id[file_id]
                        files.append((f.name, f.getvalue()))
                        overrides.append(plan_type)

                parsed = P.parse_case_files(files, form, plan_type_overrides=overrides)
                form = form.model_copy(update={"hn": parsed.resolved_hn})

                st.session_state["new_case_preview"] = dict(
                    form=form,
                    plan_frames=parsed.plan_frames,
                    warnings=parsed.warnings,
                    overwrite=bool(existing and overwrite),
                    existing_patient_id=existing.id if existing else None,
                )
            except P.UnreadableFileError as exc:
                st.error(
                    f"⚠️ That doesn't look like a valid Excel file: {exc}. Check that you "
                    "selected the RayStation clinical-goal export (.xlsx) and not a renamed, "
                    "corrupted, or password-protected file."
                )
            except P.HNMismatchError as exc:
                st.error(f"HN safeguard: {exc}")
            except P.ParserError as exc:
                st.error(f"Could not parse the upload: {exc}")

# --------------------------------------------------------------------------- #
# Preview + confirm
# --------------------------------------------------------------------------- #

preview = st.session_state.get("new_case_preview")
if preview:
    st.divider()
    form: FormInput = preview["form"]
    regimen_label = form.dose_regimen.value if form.dose_regimen else "no dose regimen set"
    st.subheader(f"{form.pt_no} — {regimen_label}")

    for w in preview["warnings"]:
        st.info(w, icon="ℹ️")

    edited_pending = render_pending_preview(preview["plan_frames"], key="new_case")

    corrected_by = ""
    if not edited_pending.empty:
        corrected_by = st.text_input(
            "Your name (required to save any correction filled in above)",
            value=st.session_state.get("entered_by", ""),
            key="new_case_corrected_by",
        )

    any_filled = (not edited_pending.empty) and edited_pending.apply(
        lambda r: pd.notna(r["value"]) or bool(r["confirmed_not_evaluable"]) or pd.notna(r["priority"]),
        axis=1,
    ).any()

    if st.button("Confirm and save", type="primary"):
        if any_filled and not corrected_by.strip():
            st.error("Enter your name before saving — at least one correction was filled in.")
        else:
            try:
                entered_by = st.session_state.get("entered_by")
                if preview["overwrite"]:
                    update_patient(engine, preview["existing_patient_id"], form, entered_by=entered_by)
                    for frame in preview["plan_frames"]:
                        replace_plan(engine, preview["existing_patient_id"], frame)
                else:
                    save_case(engine, form, preview["plan_frames"], entered_by=entered_by)

                saved, warns, errors = apply_pending_corrections(
                    engine, form.pt_no, edited_pending, corrected_by=corrected_by,
                )

                for w in warns:
                    st.warning(w)
                for e in errors:
                    st.error(e)

                verb = "Updated" if preview["overwrite"] else "Saved"
                st.success(f"{verb} case {form.pt_no} ({len(preview['plan_frames'])} plan(s))"
                          f"{f', {saved} correction(s)' if saved else ''}.")
                st.caption("Start a new case above whenever you're ready.")
                # Not calling st.rerun() here: it would immediately discard
                # this success message before the user ever saw it. Clearing
                # the preview now just means the next real interaction with
                # this page (a new Preview click, or navigating back to it)
                # starts fresh rather than reshowing this saved preview.
                del st.session_state["new_case_preview"]

            except Exception as exc:  # noqa: BLE001 — surface it rather than a blank crash
                st.error(f"Could not save this case: {exc}")
