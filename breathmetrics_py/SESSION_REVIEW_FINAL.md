# Final session review (2026-09-24)

The investigator's final, by-eye review of every session. This is the frozen
cohort definition for the final statistical test. Any session not named below
is processed exactly as the default pipeline does (D102 rest spans, automatic
trial verdicts, no override).

All values live in `breathmetrics_py/pipeline_config.py` — the single source
`final_pipeline.py` reads. Change them there, never downstream.
Times are seconds on the recording clock (the same clock as `onset_s`,
`code_time_s` and the D102 rest spans reported in `_out_final/session_qc.csv`).

## 1. Excluded sessions (`SUBJECTS_EXCLUDE`)

| Session | Status |
|---|---|
| DA01 (both) | excluded (earlier review, unchanged) |
| NB03 (both) | excluded (earlier review, unchanged) |
| EV15 (both) | excluded — confirmed in final review |
| ES32/mor | excluded — confirmed in final review |
| LG07/mor | excluded — confirmed in final review |
| OA30/mor | excluded — confirmed in final review |
| YL26/mor | excluded — confirmed in final review |
| ES29/eve, MG14/mor, MS13/eve, YR08/mor | excluded (earlier review, unchanged) |

**ER23/eve was removed from the exclusion list** — the final review rescues it
with a hand-picked rest span (section 2).

## 2. Fixation (rest) reference overrides (`MANUAL_BASELINE_SPANS`)

The fixation reference is the median over all breaths whose onset falls in the
session's rest spans (`rest_pre` ∪ `rest_post`, pooled). An override replaces a
D102 span; `post: None` removes the post block so the reference is pre-only.
"Current marker" = the D102 time the default pipeline used.

| Session | rest_pre (s) | rest_post (s) | Reference built from | Note |
|---|---|---|---|---|
| AH19/mor | D102 (193.031–253.249) | **dropped** | pre only | |
| ER23/eve | **480–580** | **dropped** | pre only | un-excluded |
| ER23/mor | **110–180** | **900–940** | pooled | |
| ES29/mor | **325–400** | D102 (1098.280–1158.494) | pooled | |
| LG07/eve | **120–200** | 869.536 (marker) – **940** | pooled | |
| LO21/mor | **150–205** | 945.184 (marker) – **1020** | pooled | |
| MG14/eve | **59–120** | D102 (835.504–895.712) | pooled | |
| ML28/mor | D102 (196.350–256.574) | **1020** – 1078.968 (marker end) | pooled | start moved, end kept |
| MS18/eve | **150–220** | 861.342 (marker) – **930** | pooled | |
| YL26/eve | 340.731 (marker) – **420** | 1087.311 (marker) – **1137** | pooled | |
| SH25/eve | 832.341–892.561 | 1581.73–1641.94 | pooled | earlier review, unchanged |
| YR08/eve | 319.78–380.0 | **dropped** | pre only | earlier review, unchanged |

Every override span lies outside the task (before the first / after the last
picture code), so no rest breath is also a trial breath.

## 3. Manual trial rejections (`MANUAL_TRIAL_REJECT`, keyed by `code_value`)

| Session | code_value | trial_index | code time | Source |
|---|---|---|---|---|
| OA30/eve | 4 | 20 | 595.65 s | final review ("trial right before t600") |
| AH19/eve | 10 | — | — | earlier review |
| AG05/mor | 11 | — | — | earlier review |
| YR08/eve | 9 | — | — | earlier review |

`MANUAL_TRIAL_ACCEPT`: AG05/mor code 26 (earlier review). `MANUAL_BAD_SPANS`:
AB22/mor 867.0–873.4 s (breaths there are dropped from every window).
Trial lists are **code_value**, not trial_index — convert before comparing.

## 4. Resulting cohort

- 40 sessions: 22 eve, 18 mor; 1462 kept trials (814 eve, 648 mor).
- **16 subjects have both sessions** (paired): AB22, AG05, AH19, AK12, AS09,
  AS31, EE10, ER23, LB17, LO21, MH20, ML28, MS18, OM16, RP06, SH25.
- 8 subjects contribute one session only: ES29/mor, ES32/eve, LG07/eve,
  MG14/eve, MS13/mor, OA30/eve, YL26/eve, YR08/eve. They cannot enter a
  paired eve-vs-mor test.
- Smallest rest pool: AH19/mor, 14 breaths.

## 5. For the agent building the final statistical test

- **The data source for every test: `breathmetrics_py/_out_final/trials_final.csv`**,
  written by `final_pipeline.py` — one row per kept trial per anchor.
  `anchor` = `code` (response = up to 3 cycles after the picture code, all
  trials) or `sound` (same, after the D110 sound trigger, sound trials only).
  Columns: `code_value`, `label`/`label_name` (subjective: Negative if the
  subject rated arousal >= 7 or valence <= 3, else Neutral), `has_sound`,
  `n_resp`, and per metric (`ti_s`, `ttot_s`, `vi_over_ti`, `mv_au_min`):
  `_resp` (response mean), `_fix` (session rest median), `_delta`, `_pct`,
  `_log`. Rows in the 09-24 run: code 814 eve / 648 mor; sound 683 eve / 541 mor.
- Alongside it in `_out_final/`: `breaths.csv` (per breath; `window` ∈
  `rest_pre`/`rest_post`/`hold`/`response`/`gap`), `trial_verdicts.csv`,
  `rest_reference.csv` (per-session rest medians, pooled/pre/post),
  `subject_medians.csv`, `session_qc.csv`, `skipped.csv`.
  Method: `PIPELINE_WALKTHROUGH.md`.
- Results already run on the 09-24 tables (plots in `figures/`):
  `eve_vs_mor_all_metrics.png` (4 metrics x valence, code anchor),
  `ti_eve_vs_mor_code_and_sound.png` (Ti; all trials / Negative / Neutral
  x code / sound), `negative_vs_neutral_within_session.png`.
- Use the pooled `<metric>_fix` column as the reference; `_fix_pre` /
  `_fix_post` are diagnostics. Where post is dropped, `_fix == _fix_pre`.
- Do not re-apply, loosen or add exclusions or rest spans, and do not re-derive
  trial validity — `trial_verdicts.csv` holds the only trial verdict, and
  `trials_final.csv` already contains kept trials only.
- Unit of the test: per-subject summary (median across trials) per session and
  valence, then paired eve vs mor across the 16 paired subjects (see
  `project_ti_metric_testing_approach` memory). Report the test statistic and p
  for every metric; do not pick metrics or subsets after seeing results.
- Every override was set by the investigator by eye on the trace and is now
  frozen; none may be revised in response to a test result.
