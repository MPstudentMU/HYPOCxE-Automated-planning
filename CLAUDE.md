# CLAUDE.md — hypocxe-platform

Working rules for Claude Code in this repository. These are binding; when a
request conflicts with one of them, stop and say so before writing code.

## 1. The manual is the specification of record

`docs/analysis_manual_th_v2.md` defines every metric, criterion, threshold and
statistical method used here. If the code and the manual disagree — or if the
manual is silent on something the code needs — **stop and ask**. Do not guess a
formula, do not "fix" the code to match an assumption, and do not update the
manual to match the code. The manual wins, and changing it is the user's call.

## 2. `engine/criteria_v0.py` is frozen

Never edit `engine/criteria_v0.py` unless the user explicitly says
**"change criteria"**. Criteria V0 is the scoring baseline that every result in
this project is traceable to; silently editing it invalidates prior analyses.
This includes base scores, priority multipliers, max-score ceilings, constraint
values and structure definitions. If a task appears to require a criteria
change, stop and explain what change would be needed.

## 3. All calculation lives in `engine/`, pages only display

- `engine/` — parsing, corrections, imputation, scoring, filtering, statistics,
  chart construction, export. Pure Python, importable and testable without
  Streamlit.
- `pages/` — layout, widgets, and rendering only. A page may call engine
  functions and pass their results to Streamlit/Plotly. A page must not contain
  a metric formula, a threshold, a p-value computation, or a data transform that
  affects a reported number.

If a page needs a new number, add the function to `engine/` and call it.

## 4. HN never appears in analysis output

The hospital number (HN) must never appear in any analysis table, chart, figure,
caption, tooltip, log line, filename or export. Analysis identifies patients by
study ID only (e.g. `Pt1`). The **only** place HN may appear is the registry
(Module 1's on-screen table and Module 9's export), and there it must be
masked by default. Module 1 may offer an explicit unmask-behind-a-checkbox
control for authorized on-screen viewing — never a default, never persisted
across a session reload, and never carried into an export, chart, or any
Module 2-5 analysis table. Module 9's export is always masked; it has no
unmask option. Before adding any column, label or export field, confirm it
carries no HN.

## 5. Plan types are exactly three

`Manual`, `Auto`, `Auto+Manual` — these exact strings, in this order, everywhere:
data model, filters, group-by keys, chart legends, table columns and exports. No
other plan type exists. Do not invent variants, abbreviations or alternate
casings; if data contains something else, treat it as a data error and surface
it in Module 8 (data review) rather than silently mapping it.

## 6. Every metric function gets a pytest test

Any function in `engine/` that produces a reported number ships with a test in
`tests/`, in the same change. Tests use hand-computed expected values (or the
pilot fixtures in `tests/fixtures/pilot/`), not values captured from the
function's own output. A metric without a test is not done.

## Conventions

- Python 3.11. Type hints on engine functions; `from __future__ import annotations`.
- Run tests with `pytest` from the repository root.
- `data/`, `uploads/` and `*.db` are gitignored — never commit patient data.
