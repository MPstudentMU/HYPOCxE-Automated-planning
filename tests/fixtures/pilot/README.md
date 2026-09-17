# Pilot fixtures

Small, synthetic RayStation-style clinical-goal exports used as parser test
inputs. They are **not** copies of real patient exports — HNs here (all
starting with `9000000`) are made up, and every ROI/dose/achieved value is
invented. Real pilot exports live outside this repository; only their
*structure* (column names, sheet-name truncation, header whitespace, the
RayStation "no priority" sentinel, the shape of a duplicate double-export,
Format A vs. Format B layout) was used to build these — see
`make_fixtures.py` in this directory, which generated them and can
regenerate them. Expected values in tests are computed by hand from the
values in these files, never captured from the parser's own output.

These files must contain no HN and no other identifying data — see
CLAUDE.md rule 4. If real pilot data is added later, keep it out of git
history (de-identify it, or reference it from outside the repo) rather than
committing it here.

| File | Exercises |
| --- | --- |
| `1clinical_goals_90000001_Manual.xlsx` | Format A, Plan+Pt columns present, priority sentinel, missing AchievedValue |
| `1clinical_goals_90000001_Auto.xlsx` | Format A, no Plan/Pt column, header whitespace, a goal shared with the Manual file (for goal_key matching) |
| `1clinical_goals_90000001_case.xlsx` | Plan type resolved from a sheet name truncated to Excel's 31-char limit, with no usable filename suffix |
| `3clinical_goals_90000003_Manual.xlsx` | Single-plan file whose Pt column disagrees with itself — resolved from the filename instead, with a warning |
| `5clinical_goals_90000005_auto.xlsx` | Lower-case filename suffix; a known, exact number of duplicate rows to drop |
| `4clinical_goals_90000004_Manual.xlsx` | Used against a deliberately wrong form HN to exercise the HN-mismatch rejection |
| `6clinical_goals_90000006_combined.xlsx` | Format B: wide `Achieved_<Plan>`/`Status_<Plan>` columns, one plan with no Status column at all |
