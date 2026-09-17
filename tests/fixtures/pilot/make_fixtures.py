"""Regenerate this directory's synthetic .xlsx fixtures. Run with:
    python tests/fixtures/pilot/make_fixtures.py

These are entirely synthetic (fake patients, fake 8-digit HNs starting with
9 — never a real hospital number) but reproduce, structurally, every edge
case observed in real RayStation pilot exports:
  - Format A: single-plan-per-file, one row per clinical goal
  - a file with a Plan column vs. one without (detect via sheet name /
    filename suffix instead)
  - Excel's 31-char sheet-name truncation ("..._Auto+Manual" -> "..._Auto+M")
  - lower-case filename suffix ("_auto")
  - a Pt column with more than one distinct value in a single-plan file
  - header whitespace
  - the RayStation "no priority" sentinel (2147483647)
  - a missing AchievedValue (NaN)
  - exact duplicate rows
  - Format B: one combined workbook, wide Achieved_<Plan>/Status_<Plan>
    columns, with one plan's Status column entirely absent
"""
from pathlib import Path

import pandas as pd

OUT = str(Path(__file__).resolve().parent)

COLS = ["Pt", "Priority", "ROI", "Goal", "GoalType", "Criteria",
        "AcceptanceLevel", "ParameterValue", "AchievedValue", "Status", "Plan"]


def row(pt, priority, roi, goal, goal_type, criteria, acc, param, achieved, status, plan):
    return dict(Pt=pt, Priority=priority, ROI=roi, Goal=goal, GoalType=goal_type,
                Criteria=criteria, AcceptanceLevel=acc, ParameterValue=param,
                AchievedValue=achieved, Status=status, Plan=plan)


# --------------------------------------------------------------------------- #
# Pt1 Manual — has Plan + Pt columns; includes the shared goal, the priority
# sentinel, and a missing AchievedValue.
# --------------------------------------------------------------------------- #
pt1_manual = [
    row(1, 1, "BODY", "D0.0% <= 4820 cGy", "DoseAtVolume", "AtMost", 4820.0, 0.0, 4802.6226, "PASS", "Manual"),
    row(1, 1, "Bladder", "D0.03cc <= 4725 cGy", "DoseAtAbsoluteVolume", "AtMost", 4725.0, 0.03, 4715.2563, "PASS", "Manual"),
    row(1, 1, "zBone", "V500cGy <= 850.00 cc", "AbsoluteVolumeAtDose", "AtMost", 850.0, 500.0, None, "FAIL", "Manual"),
    row(1, 2, "Kidney_L", "Dmean <= 1500 cGy", "AverageDose", "AtMost", 1500.0, 0.0, 1382.1, "PASS", "Manual"),
    # AcceptanceLevel/AchievedValue are volume *fractions* for VolumeAtDose
    # goals (0.95 = 95%, matching real RayStation exports — see
    # engine/parser.py's module docstring); ParameterValue is the dose
    # being queried, here 95% of Pt1's Hypo Rx (4400 cGy) = 4180 cGy.
    row(1, 1, "PTV45", "V95%Rx >= 95.0%", "VolumeAtDose", "AtLeast", 0.95, 4180.0, 0.972, "PASS", "Manual"),
    row(1, 2147483647, "Rectum_new", "Dmean <= 3000 cGy", "AverageDose", "AtMost", 3000.0, 0.0, 2890.4, "PASS", "Manual"),
]
df = pd.DataFrame(pt1_manual, columns=COLS)
with pd.ExcelWriter(f"{OUT}/1clinical_goals_90000001_Manual.xlsx") as xl:
    df.to_excel(xl, sheet_name="1clinical_goals_90000001_Man", index=False)

# --------------------------------------------------------------------------- #
# Pt1 Auto — no Plan column, no Pt column (both detected from filename/sheet
# name); header whitespace on a couple of columns; shares one goal (BODY
# D0.0%<=4820) with Manual so goal_key must match across the two files.
# --------------------------------------------------------------------------- #
pt1_auto_rows = [
    dict(**{" Priority ": 1, "ROI": "BODY", " Goal": "D0.0% <= 4820 cGy", "GoalType": "DoseAtVolume",
            "Criteria": "AtMost", "AcceptanceLevel": 4820.0, "ParameterValue": 0.0,
            "AchievedValue": 4795.1102, "Status": "PASS"}),
    dict(**{" Priority ": 1, "ROI": "Bladder", " Goal": "D0.03cc <= 4725 cGy", "GoalType": "DoseAtAbsoluteVolume",
            "Criteria": "AtMost", "AcceptanceLevel": 4725.0, "ParameterValue": 0.03,
            "AchievedValue": 4710.0, "Status": "PASS"}),
    # DoseAtVolume: AcceptanceLevel/AchievedValue are doses (cGy);
    # ParameterValue is the volume *fraction* being queried (0.95 = 95%).
    dict(**{" Priority ": 3, "ROI": "1 ITV45", " Goal": "D95.0% >= 4500 cGy", "GoalType": "DoseAtVolume",
            "Criteria": "AtLeast", "AcceptanceLevel": 4500.0, "ParameterValue": 0.95,
            "AchievedValue": 4550.2, "Status": "PASS"}),
]
df = pd.DataFrame(pt1_auto_rows)
with pd.ExcelWriter(f"{OUT}/1clinical_goals_90000001_Auto.xlsx") as xl:
    df.to_excel(xl, sheet_name="1clinical_goals_90000001_Auto", index=False)

# --------------------------------------------------------------------------- #
# Pt1 Auto+Manual — sheet name truncated to Excel's 31-char limit, exactly
# as "..._Auto+Manual" truncates to "..._Auto+M"; no Plan column, and the
# filename itself is deliberately generic so only the sheet name can supply
# the plan type.
# --------------------------------------------------------------------------- #
pt1_am = [
    row(None, 1, "BODY", "D0.0% <= 4820 cGy", "DoseAtVolume", "AtMost", 4820.0, 0.0, 4780.0, "PASS", None),
    row(None, 1, "Bladder", "D0.03cc <= 4725 cGy", "DoseAtAbsoluteVolume", "AtMost", 4725.0, 0.03, 4690.0, "PASS", None),
]
df = pd.DataFrame(pt1_am, columns=COLS).drop(columns=["Pt", "Plan"])
sheet_name = "1clinical_goals_90000001_Auto+M"  # 31 chars, mirrors Excel's own truncation
assert len(sheet_name) == 31
with pd.ExcelWriter(f"{OUT}/1clinical_goals_90000001_case.xlsx") as xl:
    df.to_excel(xl, sheet_name=sheet_name, index=False)

# --------------------------------------------------------------------------- #
# Pt3 Manual — single-plan file where the Pt column disagrees with itself
# (a copy-paste artifact): must fall back to the filename's leading digit
# and log a warning instead of trusting the column.
# --------------------------------------------------------------------------- #
pt3_manual = [
    row(3, 1, "BODY", "D0.0% <= 4700 cGy", "DoseAtVolume", "AtMost", 4700.0, 0.0, 4650.0, "PASS", "Manual"),
    row(9, 1, "Bladder", "D0.03cc <= 4600 cGy", "DoseAtAbsoluteVolume", "AtMost", 4600.0, 0.03, 4580.0, "PASS", "Manual"),
    row(3, 2, "Sigmoid", "Dmean <= 3500 cGy", "AverageDose", "AtMost", 3500.0, 0.0, 3200.0, "PASS", "Manual"),
]
df = pd.DataFrame(pt3_manual, columns=COLS)
with pd.ExcelWriter(f"{OUT}/3clinical_goals_90000003_Manual.xlsx") as xl:
    df.to_excel(xl, sheet_name="3clinical_goals_90000003_Man", index=False)

# --------------------------------------------------------------------------- #
# Pt5 Auto — no Pt/Plan column, lower-case "_auto" filename suffix, and a
# known, exact number of duplicate rows (5 unique goals, 3 of them repeated
# once more = 3 exact-duplicate rows to drop). Stands in for the real Pt5
# pilot file's much larger duplicate export (not in this repo — see
# tests/fixtures/pilot/README.md); the *behavior* under test (drop exact
# duplicate rows) is the same, only the count differs.
# --------------------------------------------------------------------------- #
pt5_unique = [
    row(None, 1, "BODY", "D0.0% <= 4750 cGy", "DoseAtVolume", "AtMost", 4750.0, 0.0, 4694.0923, "PASS", None),
    row(None, 1, "Bladder", "D0.03cc <= 4650 cGy", "DoseAtAbsoluteVolume", "AtMost", 4650.0, 0.03, 4649.4844, "PASS", None),
    row(None, 1, "Bone", "V500cGy <= 850.00 cc", "AbsoluteVolumeAtDose", "AtMost", 850.0, 500.0, None, "FAIL", None),
    row(None, 1, "Femur Head Lt", "Dmean <= 1200 cGy", "AverageDose", "AtMost", 1200.0, 0.0, 1388.4604, "FAIL", None),
    row(None, 2, "Cauda Equina", "D0.0% <= 5000 cGy", "DoseAtVolume", "AtMost", 5000.0, 0.0, 3084.4446, "PASS", None),
]
pt5_rows = pt5_unique + pt5_unique[:3]  # 3 exact duplicates, like the real double-export
df = pd.DataFrame(pt5_rows, columns=COLS).drop(columns=["Pt", "Plan"])
with pd.ExcelWriter(f"{OUT}/5clinical_goals_90000005_auto.xlsx") as xl:
    df.to_excel(xl, sheet_name="5clinical_goals_90000005_auto", index=False)

# --------------------------------------------------------------------------- #
# Pt4 — HN-mismatch fixture. Filename HN (90000004) is used directly in the
# HN-safeguard test against a *different* form HN.
# --------------------------------------------------------------------------- #
pt4_manual = [
    row(4, 1, "BODY", "D0.0% <= 4800 cGy", "DoseAtVolume", "AtMost", 4800.0, 0.0, 4750.0, "PASS", "Manual"),
]
df = pd.DataFrame(pt4_manual, columns=COLS)
with pd.ExcelWriter(f"{OUT}/4clinical_goals_90000004_Manual.xlsx") as xl:
    df.to_excel(xl, sheet_name="4clinical_goals_90000004_Man", index=False)

# --------------------------------------------------------------------------- #
# Pt6 — Format B combined workbook: wide Achieved_<Plan> columns, one plan
# (Auto+Manual) with no Status_<Plan> column at all (forces compute_status).
# --------------------------------------------------------------------------- #
pt6_wide = pd.DataFrame([
    dict(Pt=6, Priority=1, ROI="BODY", Goal="D0.0% <= 4800 cGy", GoalType="DoseAtVolume",
         Criteria="AtMost", AcceptanceLevel=4800.0, ParameterValue=0.0,
         Achieved_Manual=4790.0, Status_Manual="PASS",
         Achieved_Auto=4810.0, Status_Auto="FAIL",
         **{"Achieved_Auto+Manual": 4770.0}),
    dict(Pt=6, Priority=1, ROI="Bladder", Goal="D0.03cc <= 4700 cGy", GoalType="DoseAtAbsoluteVolume",
         Criteria="AtMost", AcceptanceLevel=4700.0, ParameterValue=0.03,
         Achieved_Manual=4690.0, Status_Manual="PASS",
         Achieved_Auto=4650.0, Status_Auto="PASS",
         **{"Achieved_Auto+Manual": 4712.0}),
])
with pd.ExcelWriter(f"{OUT}/6clinical_goals_90000006_combined.xlsx") as xl:
    pt6_wide.to_excel(xl, sheet_name="6clinical_goals_90000006_comb", index=False)

print("done")
