"""Patient Data — Module 1.

The on-screen registry: one row per patient, HN masked by default (unmask
behind an explicit checkbox — never a default, never carried into a chart,
export, or any Module 2-5 analysis table; see CLAUDE.md rule 4). Filters,
cohort counts, two small bar charts, and — since a case can now be saved
with only pt_no and plan files (engine.schemas.FormInput) — an Edit
workflow to fill in whatever was left blank at intake. Display only:
engine.export builds the table (including profile_complete) and
engine.charts builds the figures; engine.storage does every write.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.db import get_engine
from components.pending_banner import render_pending_banner
from engine.charts import cases_per_planner_chart, cases_per_regimen_chart
from engine.config import DOSE_REGIMEN_UNSET, dose_regimen_choices, rx_for_dose_regimen
from engine.export import load_registry_frame, mask_hn
from engine.schemas import FormInput, PlanType
from engine.storage import find_patient_by_pt_no, get_plan_times, update_patient, update_plan_times

st.title("Patient Data")
st.caption("Module 1 — patient and plan records")

engine = get_engine()
render_pending_banner(engine)

just_edited = st.session_state.pop("patient_data_just_edited", None)
if just_edited:
    st.success(f"Updated {just_edited}.")

df = load_registry_frame(engine)
if df.empty:
    st.info("No cases yet — add one in New Case (Module 0).")
    st.stop()

n_incomplete = int((~df["profile_complete"]).sum())
if n_incomplete:
    patient_word = "patient" if n_incomplete == 1 else "patients"
    st.info(
        f"📋 {n_incomplete} {patient_word} with incomplete profile — use **Edit** below to "
        "fill in the rest.",
        icon="📋",
    )

st.subheader("Filter")
f1, f2, f3, f4, f5 = st.columns(5)
regimens = f1.multiselect("Dose regimen", sorted(df["dose_regimen"].dropna().unique()))
sib_choice = f2.selectbox("SIB boost", ["All", "Yes", "No"])
rooms = f3.multiselect("Tx room", sorted(df["tx_room"].dropna().unique()))
ros = f4.multiselect("RO", sorted(df["ro"].dropna().unique()))
mp1s = f5.multiselect("MP1", sorted(df["mp1"].dropna().unique()))

filtered = df
if regimens:
    filtered = filtered[filtered["dose_regimen"].isin(regimens)]
if sib_choice != "All":
    filtered = filtered[filtered["sib_boost"] == (sib_choice == "Yes")]
if rooms:
    filtered = filtered[filtered["tx_room"].isin(rooms)]
if ros:
    filtered = filtered[filtered["ro"].isin(ros)]
if mp1s:
    filtered = filtered[filtered["mp1"].isin(mp1s)]

st.divider()

c1, c2, c3 = st.columns(3)
c1.metric("Cases", len(filtered))
c2.metric("With SIB boost", int(filtered["sib_boost"].fillna(False).sum()))
c3.metric("Goals pending review", int(filtered["pending_count"].sum()))

st.divider()

unmask = st.checkbox(
    "Unmask HN",
    help="Shows the full HN on screen for this view only — never saved, exported, "
        "charted, or carried into any analysis table.",
)


def _dash(v) -> str:
    return "—" if pd.isna(v) else str(v)


display = filtered.copy()
display["HN"] = display["hn"] if unmask else display["hn"].map(mask_hn)
display["Straight-pass"] = display["straight_pass"].map(
    lambda v: "🔁 Yes" if v is True else ("No" if v is False else "—")
)
display["Alert"] = display["pending_count"].map(
    lambda n: "✅ OK" if n == 0 else f"⚠️ {n} pending"
)
display["Profile"] = display["profile_complete"].map(
    lambda complete: "✅ Complete" if complete else "🟠 Incomplete"
)
for col in ("dose_regimen", "tx_room", "mp1", "mp2", "ro"):
    display[col] = display[col].map(_dash)
display = display.rename(columns={
    "pt_no": "Patient", "dose_regimen": "Regimen", "sib_boost": "SIB",
    "tx_room": "Tx room", "mp1": "MP1", "mp2": "MP2", "ro": "RO",
    "plans_uploaded": "Plans uploaded", "created_at": "Upload date",
})

st.dataframe(
    display[["Patient", "HN", "Regimen", "SIB", "Tx room", "MP1", "MP2", "RO",
             "Plans uploaded", "Straight-pass", "Profile", "Alert", "Upload date"]],
    hide_index=True,
    width="stretch",
)

st.divider()

st.subheader("Cohort")
chart_col1, chart_col2 = st.columns(2)
chart_col1.plotly_chart(cases_per_regimen_chart(filtered), width="stretch")
chart_col2.plotly_chart(cases_per_planner_chart(filtered), width="stretch")

# --------------------------------------------------------------------------- #
# Edit — fill in (or correct) a patient's intake details after the fact
# --------------------------------------------------------------------------- #


@st.dialog("Edit patient", width="large")
def _edit_patient_dialog(pt_no: str) -> None:
    patient = find_patient_by_pt_no(engine, pt_no)
    if patient is None:
        st.error("This patient no longer exists.")
        return

    st.subheader(pt_no)
    hn = st.text_input("HN", value=patient.hn)

    dose_choices = dose_regimen_choices()
    options = [DOSE_REGIMEN_UNSET, *dose_choices.keys()]
    current_label = next(
        (label for label, regimen in dose_choices.items() if regimen == patient.dose_regimen),
        DOSE_REGIMEN_UNSET,
    )
    dose_choice = st.radio("Dose regimen", options, index=options.index(current_label))
    sib_boost = st.checkbox("SIB boost", value=bool(patient.sib_boost))

    ec1, ec2 = st.columns(2)
    with ec1:
        tx_room = st.text_input("Treatment room", value=patient.tx_room or "")
        mp1 = st.text_input("Medical physicist 1", value=patient.mp1 or "")
    with ec2:
        mp2 = st.text_input("Medical physicist 2", value=patient.mp2 or "")
        ro = st.text_input("Radiation oncologist", value=patient.ro or "")

    st.markdown("**Planning time (minutes)**")
    current_times = get_plan_times(engine, patient.id)
    new_times: dict[PlanType, float | None] = {}
    if current_times:
        ordered = [pt for pt in (PlanType.MANUAL, PlanType.AUTO, PlanType.AUTO_MANUAL)
                  if pt in current_times]
        time_cols = st.columns(len(ordered))
        for col, plan_type in zip(time_cols, ordered):
            entered = col.number_input(
                plan_type.value, min_value=0.0, step=1.0,
                value=float(current_times[plan_type] or 0.0),
            )
            new_times[plan_type] = entered or None
    else:
        st.caption("No plans uploaded yet — nothing to set a planning time for.")

    if st.button("Save", type="primary"):
        if not hn.strip():
            st.error("HN is required — it can't be cleared, only corrected.")
            return

        dose_regimen = dose_choices.get(dose_choice)  # None if left unset
        rx_cgy, fractions = rx_for_dose_regimen(dose_regimen) if dose_regimen else (None, None)
        form = FormInput(
            hn=hn.strip(),
            pt_no=pt_no,
            dose_regimen=dose_regimen,
            rx_cgy=rx_cgy,
            fractions=fractions,
            sib_boost=sib_boost,
            tx_room=tx_room.strip() or None,
            mp1=mp1.strip() or None,
            mp2=mp2.strip() or None,
            ro=ro.strip() or None,
        )
        update_patient(engine, patient.id, form, entered_by=st.session_state.get("entered_by"))
        if new_times:
            update_plan_times(engine, patient.id, new_times)
        # st.rerun() is what closes the dialog and refreshes the Manage
        # patients table/badges underneath (dismissing an st.dialog does
        # NOT rerun the page by itself — on_dismiss defaults to "ignore").
        # A plain st.success() right before it would flash and vanish
        # before anyone saw it, so the confirmation is stashed in
        # session_state and shown on the page itself after the rerun
        # completes — same pattern pages/0_new_case.py uses for its own
        # "Confirm and save" success message.
        st.session_state["patient_data_just_edited"] = pt_no
        st.rerun()


st.divider()

st.subheader("Manage patients")
st.caption("Fill in or correct a patient's details — files/goals themselves are re-uploaded "
          "from New Case, not edited here.")
for _, row in filtered.iterrows():
    rc1, rc2, rc3 = st.columns([2, 4, 1])
    rc1.markdown(f"**{row['pt_no']}**")
    with rc2:
        if not row["profile_complete"]:
            st.markdown(":orange-badge[🟠 Incomplete profile]")
        if row["pending_count"]:
            st.markdown(f":red-badge[⚠️ {row['pending_count']} goal(s) pending]")
    if rc3.button("Edit", key=f"edit_{row['pt_no']}"):
        _edit_patient_dialog(row["pt_no"])
