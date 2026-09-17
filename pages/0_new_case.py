"""New Case — Module 0.

Intake: fill the case form, upload the plan export file(s), preview what
was parsed (goal counts, warnings, anything needing a correction), then
confirm to save. All parsing, validation and persistence lives in engine/;
this file only renders the form, the preview, and the editor.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from components.pending_preview import render_pending_preview
from engine import corrections as C
from engine import parser as P
from engine.config import rx_for_dose_regimen
from engine.schemas import (
    CorrectionField,
    CorrectionSource,
    CorrectionStatus,
    DoseRegimen,
    FormInput,
    PlanType,
)
from engine.storage import find_patient_by_pt_no, load_analysis_frame, replace_plan, save_case, update_patient

st.title("New Case")
st.caption("Module 0 — intake of a new planning case")

engine = get_engine()
render_pending_banner(engine)

DOSE_CHOICES = {
    f"Hypo — {rx_for_dose_regimen(DoseRegimen.HYPO)[0]/100:g} Gy / "
    f"{rx_for_dose_regimen(DoseRegimen.HYPO)[1]} fx": DoseRegimen.HYPO,
    f"Conv — {rx_for_dose_regimen(DoseRegimen.CONV)[0]/100:g} Gy / "
    f"{rx_for_dose_regimen(DoseRegimen.CONV)[1]} fx": DoseRegimen.CONV,
}

with st.form("intake_form"):
    st.subheader("Case details")
    c1, c2 = st.columns(2)
    with c1:
        hn = st.text_input("HN", help="Leave blank to use the value from the uploaded filenames.")
        pt_no = st.text_input("Patient no.", placeholder="Pt7")
        dose_choice = st.radio("Dose regimen", list(DOSE_CHOICES.keys()))
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
    st.caption("RayStation clinical-goal export — one file per plan.")
    f1, f2, f3 = st.columns(3)
    manual_file = f1.file_uploader("Manual", type=["xlsx"], key="upload_manual")
    auto_file = f2.file_uploader("Auto", type=["xlsx"], key="upload_auto")
    automanual_file = f3.file_uploader(
        "Auto+Manual", type=["xlsx"], key="upload_automanual",
        help="Optional — leave empty if the Auto plan needed no manual touch-up (a "
            "'straight pass'). Its Auto plan results are then used for Auto+Manual "
            "in the analysis instead.",
    )

    preview_clicked = st.form_submit_button("Preview")

if preview_clicked:
    if not pt_no.strip():
        st.error("Enter a patient number.")
    elif not manual_file or not auto_file:
        st.error("Both the Manual and Auto plan files are required (Auto+Manual is optional).")
    else:
        existing = find_patient_by_pt_no(engine, pt_no.strip())
        if existing and not overwrite:
            st.error(
                f"Patient no. '{pt_no.strip()}' already exists. Check "
                "'Overwrite if this patient no. already exists' to replace it."
            )
        else:
            rx_cgy, fractions = rx_for_dose_regimen(DOSE_CHOICES[dose_choice])
            try:
                form = FormInput(
                    hn=hn.strip() or None,
                    pt_no=pt_no.strip(),
                    dose_regimen=DOSE_CHOICES[dose_choice],
                    rx_cgy=rx_cgy,
                    fractions=fractions,
                    sib_boost=sib_boost,
                    tx_room=tx_room.strip(),
                    mp1=mp1.strip(),
                    mp2=mp2.strip(),
                    ro=ro.strip(),
                    time_manual=time_manual or None,
                    time_auto=time_auto or None,
                    time_automanual=time_automanual or None,
                )
                files = [(manual_file.name, manual_file.getvalue()),
                        (auto_file.name, auto_file.getvalue())]
                overrides = [PlanType.MANUAL, PlanType.AUTO]
                if automanual_file:
                    files.append((automanual_file.name, automanual_file.getvalue()))
                    overrides.append(PlanType.AUTO_MANUAL)

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
    st.subheader(f"{form.pt_no} — {form.dose_regimen.value}")

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

                saved, errors, warns = 0, [], []
                if not edited_pending.empty:
                    df = load_analysis_frame(engine)
                    df = df[df["pt_no"] == form.pt_no]
                    goal_id_by_key = {
                        (r.plan_type, r.goal_key): r.goal_id for r in df.itertuples()
                    }
                    for _, row in edited_pending.iterrows():
                        goal_id = goal_id_by_key.get((row["plan_type"], row["goal_key"]))
                        if goal_id is None:
                            continue
                        label = f"{row['plan_type']} / {row['roi']}"
                        source = CorrectionSource(row["source"]) if row["source"] else None

                        if row["_missing_value"] and (row["confirmed_not_evaluable"] or pd.notna(row["value"])):
                            try:
                                status = (CorrectionStatus.CONFIRMED_NOT_EVALUABLE
                                         if row["confirmed_not_evaluable"] else CorrectionStatus.CORRECTED)
                                _, warn = C.record_correction(
                                    engine, goal_id=goal_id, field=CorrectionField.ACHIEVED_VALUE,
                                    status=status, source=source, reason=row["reason"],
                                    corrected_by=corrected_by,
                                    corrected_value_display=(
                                        None if row["confirmed_not_evaluable"] else float(row["value"])
                                    ),
                                )
                                saved += 1
                                if warn:
                                    warns.append(f"{label}: {warn}")
                            except C.CorrectionValidationError as exc:
                                errors.append(f"{label}: {exc}")

                        if row["_missing_priority"] and pd.notna(row["priority"]):
                            try:
                                C.record_correction(
                                    engine, goal_id=goal_id, field=CorrectionField.PRIORITY,
                                    status=CorrectionStatus.CORRECTED, source=source,
                                    reason=row["reason"], corrected_by=corrected_by,
                                    corrected_value_display=row["priority"],
                                )
                                saved += 1
                            except C.CorrectionValidationError as exc:
                                errors.append(f"{label}: {exc}")

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
