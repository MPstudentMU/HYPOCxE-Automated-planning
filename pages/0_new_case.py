"""New Case — Module 0.

Intake: fill the case form, upload the plan export file(s), save, then
resolve anything the upload flagged as needing a manual correction. All
parsing, validation and persistence lives in engine/; this file only
renders the form, the preview table, and the correction editor.
"""
from __future__ import annotations

import streamlit as st

from components.correction_editor import render_correction_editor
from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine import corrections as C
from engine import parser as P
from engine.schemas import DoseRegimen, FormInput
from engine.storage import find_patient_by_pt_no, load_analysis_frame, replace_plan, save_case, update_patient

st.title("New Case")
st.caption("Module 0 — intake of a new planning case")

engine = get_engine()
render_pending_banner(engine)

with st.form("intake_form"):
    st.subheader("Case details")
    c1, c2 = st.columns(2)
    with c1:
        hn = st.text_input("HN", help="Leave blank to use the value from the uploaded filenames.")
        pt_no = st.text_input("Patient no.", placeholder="Pt7")
        dose_regimen = st.selectbox("Dose regimen", [d.value for d in DoseRegimen])
        rx_cgy = st.number_input("Prescription dose (cGy)", min_value=0.0, step=100.0)
        fractions = st.number_input("Fractions", min_value=1, step=1)
        sib_boost = st.checkbox("SIB boost")
    with c2:
        tx_room = st.text_input("Treatment room")
        mp1 = st.text_input("Medical physicist 1")
        mp2 = st.text_input("Medical physicist 2")
        ro = st.text_input("Radiation oncologist")

    st.subheader("Planning time (minutes)")
    t1, t2, t3 = st.columns(3)
    time_manual = t1.number_input("Manual", min_value=0.0, step=1.0, value=0.0)
    time_auto = t2.number_input("Auto", min_value=0.0, step=1.0, value=0.0)
    time_automanual = t3.number_input("Auto+Manual", min_value=0.0, step=1.0, value=0.0)

    st.subheader("Plan export files")
    st.caption("RayStation clinical-goal exports (one file per plan) or a combined workbook.")
    uploads = st.file_uploader("Upload file(s)", type=["xlsx"], accept_multiple_files=True)

    submitted = st.form_submit_button("Save case")

if submitted:
    if not uploads:
        st.error("Upload at least one plan export file.")
    elif not pt_no.strip():
        st.error("Enter a patient number.")
    else:
        try:
            form = FormInput(
                hn=hn.strip() or None,
                pt_no=pt_no.strip(),
                dose_regimen=DoseRegimen(dose_regimen),
                rx_cgy=rx_cgy,
                fractions=int(fractions),
                sib_boost=sib_boost,
                tx_room=tx_room.strip(),
                mp1=mp1.strip(),
                mp2=mp2.strip(),
                ro=ro.strip(),
                time_manual=time_manual or None,
                time_auto=time_auto or None,
                time_automanual=time_automanual or None,
            )
            parsed = P.parse_case_files([(f.name, f.getvalue()) for f in uploads], form)
            form = form.model_copy(update={"hn": parsed.resolved_hn})

            existing = find_patient_by_pt_no(engine, form.pt_no)
            if existing:
                update_patient(engine, existing.id, form)
                for frame in parsed.plan_frames:
                    replace_plan(engine, existing.id, frame)
                st.session_state["new_case_pt_no"] = form.pt_no
                st.success(f"Updated existing case {form.pt_no} ({len(parsed.plan_frames)} plan(s)).")
            else:
                save_case(engine, form, parsed.plan_frames)
                st.session_state["new_case_pt_no"] = form.pt_no
                st.success(f"Saved new case {form.pt_no} ({len(parsed.plan_frames)} plan(s)).")

            for w in parsed.warnings:
                st.info(w, icon="ℹ️")

        except P.HNMismatchError as exc:
            st.error(f"HN safeguard: {exc}")
        except P.ParserError as exc:
            st.error(f"Could not parse the upload: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface it rather than a blank crash
            st.error(f"Could not save this case: {exc}")

# --------------------------------------------------------------------------- #
# Preview: goals from the case just saved that still need a correction
# --------------------------------------------------------------------------- #

pt_no_to_review = st.session_state.get("new_case_pt_no")
if pt_no_to_review:
    st.divider()
    st.subheader(f"Review — {pt_no_to_review}")
    df = load_analysis_frame(engine)
    patient_df = df[df["pt_no"] == pt_no_to_review]
    if patient_df.empty:
        st.info("No goals found for this case.")
    else:
        corrected_df = C.apply_corrections(patient_df, engine)
        render_correction_editor(engine, corrected_df, key="new_case")
