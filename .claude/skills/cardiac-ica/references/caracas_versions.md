# CARACAS versions, repos and dates

Re-verify with `scripts/check_upstream.sh` — this table was captured 2026-09-30 on the previous Mac setup.

**This machine (2026-10-01):** only `D:\user\Desktop\startle-repo\SASICA\` (`9ab76d7`, = `SASICA_DIR`) and `Cardiac_IC_labelling\` exist; `SASICA\CARACAS\heart_functions` is empty and there is no `~/code/tools/`. The `~/code/tools/...` rows below don't exist here.

## Repos on disk (Mac, 2026-09-30)

| Clone | Remote | HEAD (date) | Newest upstream | Behind |
|---|---|---|---|---|
| `hr/SASICA/` (untracked in hr repo) | github.com/dnacombo/SASICA, branch `master` | `9ab76d7` 2026-05-22 | same | 0 |
| `~/code/tools/SASICA/` (= `SASICA_DIR` in `hr/config.py`; used by `hr/caracas/caracas_session.m`) | same | `9ab76d7` 2026-05-22 | same | 0 |
| `~/code/tools/SASICA/CARACAS/heart_functions` (submodule) | github.com/dnacombo/heart_functions | `0db5c42` 2025-07-25 (pinned by SASICA) | `194e35d` 2026-05-04 | 7 — Python port + license only; `heart_peak_detect.m` identical |
| `hr/SASICA/CARACAS/heart_functions` | — | **empty** (submodule not initialized) | | |
| `~/code/tools/heart_functions/` (= `HEART_FUNCTIONS_DIR`) | heart_functions | `194e35d` 2026-05-04 (shallow) | same | 0 |
| `hr/Cardiac_IC_labelling/` | github.com/PierreChampetier/Cardiac_IC_labelling, `main` | `bb41378` 2025-01-21 | branch `Test_heart_functions` `671a1a9` 2025-02-04 | 0 on main |

SASICA branches: `master` has CARACAS merged (PR #15, 2026-05-22). `origin/CARACAS` stops at `b47e561` (same day, README differs only). The GitHub URL `tree/CARACAS` in the project docs therefore points to a branch that is now *behind* master by one README commit — master is the current code.

## Timeline of the CARACAS method

- 2024-08 → 2025-01-15 — Champetier's original (`Cardiac_IC_labelling`), final form "Selection of cardiac IC based on QRSampl and RR only (5 parameters)" (`786beeb`).
- 2025-01-21 — Chaumon PR adds heart_functions path to Champetier's repo (`df5ace1`); Champetier's `Test_heart_functions` branch (2025-02) swaps the detector to `heart_peak_detect(cfg_peak, comp_continu)` — the precursor of version B.
- 2025-07-25 — heart_functions gains `abstemplate` and `NaNST` options (`0db5c42`, the commit SASICA pins).
- 2025-11-12 → 2026-05-21 — CARACAS integrated into `eeg_SASICA.m`, RPeakstoNoise added (2026-01-09), NaN padding between trials (2026-02-04), thresholds re-optimized five times (see `caracas_sasica.md` §6).
- 2026-05-22 — merged to SASICA master; `CARACAS/CARACAS.m` (version B) and docs added the same day.

## Version B — `SASICA/CARACAS/CARACAS.m`

Signature: `[heart_IC, meas, aaa_parameters_find_heart_IC, output_for_zscore_corMatrix_ROC, output_for_user] = CARACAS(cfg, comp)` with `comp` a FieldTrip component struct. Header comments still describe version C's usage (`A_fct_find_cardiac_IC`, method 1/2) — ignore them.

- Requires **continuous data** (`numel(comp.trial) > 1` → error).
- Calls `heart_peak_detect(cfg_peak, comp)` with `cfg_peak.channel = comp.label{i}`, `corthresh = 0.2`, `absPT = 0`.
- Measures: `PQ, QS, ST, PR, RT, PT` interval CVs (15–85 pct trimmed), `sk`, `ku`, `RR` CV (0–70 pct), `Rampl` CV (15–85), `Ampl_var` (range-of-range across `mini_bouts_duration_for_SignalAmplRange` = 10 s bouts), `bpm`.
- bpm duration subtracts 0.5 s per NaN block (edge-effect allowance).
- Decision: bpm in [`bpm_min`=35, `bpm_max`=90] AND RR CV ≤ 1/3 AND sk ≥ 2. Rampl, PQ, Ampl_var, ku, `method_chosen`, `nb_IC_wanted`, `threshold_cond_IC_method1` (0.6), `threshold_std_method2` (2.5), `threshold_regularity_signal_minmax` (1.5) are parsed but **not used** in the decision (the corresponding lines are commented out).
- `min_recording_duration_sec` = 20: shorter → warning, `heart_IC = []`, and `meas` is left unassigned (caller asking for it errors).
- `IC_to_not_analyze` list → skipped ICs, but `meas` isn't filled for them.
- `plot_heart_IC = 1` needs `path_output`, `file_info`, and helpers `timepts`, `ifelse`, `mymkdir` that are **not on any path here** → errors.
- `rep2struct` required (shim provides it; SASICA's `private/rep2struct.m` is not visible from `CARACAS/`).
- `output_for_user.heart_IC_nbr_cardiac_event` uses a 3-SD event-count threshold unrelated to the decision — don't interpret it.

Used by `hr/matlab/caracas_run.m`.

## Version C — `Cardiac_IC_labelling/CARACAS.m` (Champetier original)

Signature: `[aaa_parameters_find_heart_IC, output_for_zscore_corMatrix_ROC, output_for_user] = CARACAS(cfg, comp)`. The README calls it `A_fct_find_cardiac_IC` and lists `[rejected_heart_IC, table_heart_IC, method_reject_cardiac_IC]` outputs — stale; the result is `output_for_user.heart_IC`.

- Detector: `ECG_PQRST_VERSION_3_PC` (Sanghavi 2021, MATLAB File Exchange #73850): fixed 55-tap low-pass FIR `filter` (causal, adds delay), `baseline_remove`, normalize, **drop first 99 samples**, Pan-Tompkins (`pan_tompkin2`), then Q/S/P/T search windows; Champetier's edits marked `PIERRE`.
- Features per IC: QRS peak-to-peak "ERP" amplitude median/skew/std÷median (±0.2 s windows), RR CV and RR skew, plus Q/S/R amplitude stats and χ² uniformity (computed, mostly unused).
- Method 1 (`absolute_threshold`, default): an IC scores 1 per feature if it's in the lowest `nb_IC_wanted`(3)-of-N percentile across ICs; score = mean over 5 features; cardiac iff score ≥ `threshold_cond_IC_method1` (code default **0.6**, README says 0.5) AND 45 < bpm < 90 AND amplitude-regularity (max−min)/min of 10 s bout means < 1.5.
- Method 2 (`mean_std`): score > mean + 2.5·SD of all IC scores, same bpm/regularity gates.
- **Relative ranking** → result depends on how many ICs there are; with few ICs, "top 3" is a large fraction.
- ICs with < 10 R peaks get NaN features.
- Needs `skewness`, `kurtosis`, `prctile`, `nanzscore` (local), `histcounts`, `detrend(x,10)`, and **`chi2inv` (Statistics Toolbox — not installed here → C cannot run on this machine as is)**.

## When someone says "CARACAS"

Ask/confirm which one when it matters. Default meaning in this repo going forward: version A (the `eeg_SASICA` block), because it is the maintained, merged implementation with the current thresholds. Flag to the user if a script uses B or C.
