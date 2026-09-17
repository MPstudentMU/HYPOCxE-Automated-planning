# hypocxe-platform

A Streamlit platform for analysing hypofractionated radiotherapy treatment
plans — comparing **Manual**, **Auto** and **Auto+Manual** planning across pass
rate, planning time, plan quality and DVH metrics.

## Status

Project scaffold. Module 4 (plan quality with priority filtering) has a first
implementation; the remaining pages are placeholders.

## Requirements

- Python 3.11

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

## Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Navigation shell |
| `pages/` | UI only — layout and rendering |
| `engine/` | All calculation: parsing, scoring, stats, charts, export |
| `components/` | Reusable UI fragments |
| `reference/` | Lookup data (ROI aliases) |
| `docs/` | `analysis_manual_th_v2.md` — the specification of record |
| `tests/` | pytest suite, pilot fixtures |
| `data/` | Local database and uploads (gitignored) |

## Modules

| Page | Module |
| --- | --- |
| `0_new_case.py` | New case intake |
| `1_patient_data.py` | Patient data |
| `2_pass_rate.py` | Pass rate |
| `3_time_efficiency.py` | Time efficiency |
| `4_plan_quality.py` | Plan quality score |
| `5_dvh_comparison.py` | DVH comparison |
| `8_data_review.py` | Data review |
| `9_registry_export.py` | Registry export |

## Ground rules

See `CLAUDE.md`. In short: the manual in `docs/` is authoritative,
`engine/criteria_v0.py` is frozen, calculation lives in `engine/` and pages only
display, HN never leaves the registry, plan types are exactly the three above,
and every metric function has a test.
