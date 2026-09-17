# hypocxe-platform

A Streamlit platform for analysing hypofractionated radiotherapy treatment
plans — comparing **Manual**, **Auto** and **Auto+Manual** planning across pass
rate, planning time, plan quality and DVH metrics.

## Status

All nine modules are implemented: intake (0), patient registry (1), pass
rate (2), time efficiency (3), plan quality with priority filtering (4), DVH
comparison (5), data review/corrections (8), and registry export (9).

## Requirements

- Python 3.11

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The app is gated behind a shared passphrase (see
[Access control](#access-control)) and won't start without one configured:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python -c "import hashlib; print(hashlib.sha256(b'yourpassphrase').hexdigest())"
# paste the printed hash into .streamlit/secrets.toml as app_passphrase_hash
```

## Running locally

```bash
streamlit run app.py
```

This starts a local web server (default `http://localhost:8501`) backed by a
SQLite database at `data/hypocxe.db`, created automatically on first run. The
app is meant to run on one machine on the clinic's local network — see
[PDPA / HN handling](#pdpa--hn-handling) and
[Access control](#access-control) below for why.

> If you have an existing `data/hypocxe.db` from before the `entered_by`
> column was added to `patients`/`analysis_runs` (see Access control),
> delete it (or back it up and start fresh) before running the app again —
> `engine.storage.init_db` only creates missing *tables*, not missing
> *columns* on a table that already exists, so an old file will raise
> `sqlite3.OperationalError: no such column` rather than silently working.

On every process start, `app.py` makes a same-day backup of `data/hypocxe.db`
into `data/backups/` (one file per calendar day; a no-op if today's backup
already exists — see `engine.storage.ensure_daily_backup`). This is a
same-machine safety net against an accidental delete or corruption during a
session, not a substitute for your own off-machine backup of `data/`.

The sidebar shows a short recent-activity log (new cases, corrections,
exports) sourced from the same tables `engine.storage.recent_activity`
already logs to — nothing extra to configure.

## Test

```bash
pytest
```

## Access control

The whole app sits behind one shared passphrase (`engine/auth.py`, gated in
`app.py` before `st.navigation` runs) — not per-user accounts:

- **Setup.** `.streamlit/secrets.toml` (gitignored — never commit it) holds
  only `app_passphrase_hash`, the SHA-256 hash of the passphrase; the
  plaintext value itself is never stored or logged anywhere. See
  `.streamlit/secrets.toml.example` for the exact command to generate a hash,
  and the [Setup](#setup) section above.
- **Rotating the passphrase.** Generate a new hash with the same command and
  replace `app_passphrase_hash` in `.streamlit/secrets.toml` — every session
  already logged in stays logged in (nothing checks the hash again mid-
  session) but a fresh login needs the new passphrase.
- **"Entered by."** Right after a successful login, a one-time prompt asks
  who's using the app this session. It's stored only in that session's
  `st.session_state["entered_by"]` — never a real account, never persisted
  once the session ends — and is: (a) stamped onto `patients.entered_by`
  when a case is saved and onto `analysis_runs.entered_by`/the export's
  `RunInfo` sheet when a workbook is built, and (b) used to pre-fill (not
  force-overwrite — it can still be edited) the "Your name" field wherever a
  correction is recorded.
- **What this is not.** A shared passphrase means anyone who has it can act
  as anyone — there's no way to tell two people apart from the login alone
  (that's what "Entered by" is for, on a good-faith basis, not an audit
  guarantee). This is suitable **only** for a trusted-network deployment —
  see PDPA / HN handling below — not a substitute for per-user accounts with
  real authentication. If the clinic ever needs to know for certain who did
  what, this isn't that.

## PDPA / HN handling

The hospital number (HN) is the one field in this app that's directly
identifying, so it's handled narrowly on purpose:

- **Local network only.** The [passphrase gate](#access-control) above is a
  shared-secret speed bump, not real per-user authentication — run this on a
  machine/network the clinic already controls access to (the hospital
  network), the same way you would a shared spreadsheet on a department
  drive. Don't expose the Streamlit server to the public internet.
- **Masked by default.** Every on-screen table shows HN masked
  (`engine.export.mask_hn` — all but the last 4 digits) except Module 1
  (Patient Data), which has one explicit "Unmask HN" checkbox — off by
  default, never persisted, and scoped to that one view.
- **Never exported.** Module 9's workbook (`engine/export.py`,
  `build_analysis_workbook`) masks HN in every sheet with no unmask option —
  a file that's meant to leave the app never carries the unmasked value. Every
  other analysis table/chart (Modules 2–5, 8) excludes HN entirely rather
  than masking it (`engine.storage.load_analysis_frame` asserts this); it's
  identified only by `pt_no` (e.g. "Pt7"), which is not itself linkable to
  the patient outside this database.

This is enforced in code, not just convention — see CLAUDE.md rule 4 and
`tests/test_export.py`/`tests/test_audit_and_backup.py` for the tests that
check HN never leaks into an export, chart, or audit log.

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
