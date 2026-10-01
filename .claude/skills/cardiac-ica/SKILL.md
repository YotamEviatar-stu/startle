---
name: cardiac-ica
description: Expert on EEG, ICA, and recovering the heart (ECG) signal from EEG independent components with SASICA/CARACAS and heart_functions — including the exact versions, entry points, configuration, runtime requirements, and known bugs of the local Cardiac_IC_labelling/, SASICA/ and heart_functions clones. Use this whenever the user works on cardiac/ECG-like ICs, CARACAS, SASICA, heart_peak_detect, eeg_SASICA, ft_SASICA, ICLabel "heart beat" components, R-peak/PQRST detection on an IC, choosing ICA settings (number of components, filtering, rank) so a cardiac IC appears, or debugging why a session has no cardiac IC or a wrong bpm — even if they don't name the toolbox.
---

# Cardiac ICA expert (EEG → ICA → cardiac IC → beats)

You are the domain expert for recovering an ECG-like signal from EEG with ICA and selecting the cardiac IC with SASICA's CARACAS method. The HR repo's `.claude/CLAUDE.md` rules still bind you (sourced methods only, no new rejection gates, blinding, no SpO2 cross-checks, cardiac scope only). This skill tells you *what the toolkits actually do*, so you can run them correctly and explain them precisely — not license to invent methods around them.

## 1. Which code is authoritative (check this first)

There are **three different implementations named "CARACAS"** on disk. They give different answers for the same data. Know which one you are talking about.

| # | Where | Date | Detector | Decision rule | Status |
|---|---|---|---|---|---|
| A | `SASICA/eeg_SASICA.m` lines 699–795 (the `cfg.CARACAS` block) | thresholds finalized 2026-05-21 (`1aade2a`), merged to master 2026-05-22 (`b100ebc`) | `heart_peak_detect` (heart_functions) | IC is cardiac iff it passes **all 5**: sk ≥ 1.0, ku ≥ 5.3, RR-CV ≤ 0.31, Rampl-CV ≤ 0.23, 34 ≤ bpm ≤ 97 | **Current, maintained. Use this.** |
| B | `SASICA/CARACAS/CARACAS.m` (643 lines) | file added 2026-05-22 (`79d9bee`) but content is an older intermediate version | `heart_peak_detect` (FieldTrip `cfg,comp` call) | bpm in [35, 90] AND RR-CV ≤ 1/3 AND sk ≥ 2 | Legacy snapshot. Header/usage text still describes the original version. Don't treat as the reference. |
| C | `Cardiac_IC_labelling/CARACAS.m` (1273 lines, Champetier) | last commit 2025-01-21 (`bb41378`) | ECG_PQRST_VERSION_3_PC (Sanghavi, Pan-Tompkins) | ranks ICs on 5 features (top `nb_IC_wanted`), score ≥ 0.6 + bpm [45, 90] + amplitude regularity | Original research code. Superseded. |

The user decided (2026-09-30) that stage 1 is built on **A**; the project `CLAUDE.md` says so. `hr/caracas/caracas_session.m` calls A. `hr/matlab/caracas_run.m` still calls B — flag it if it comes up rather than using it.

Before relying on any version claim, run `bash .claude/skills/cardiac-ica/scripts/check_upstream.sh` — it fetches all remotes and prints HEAD date, newest remote commit, and how far behind each clone is. Upstream moves (6 CARACAS threshold changes between 2025-11 and 2026-05).

Details of each version: `references/caracas_versions.md`.

## 2. How the current method (A) works — the short version

For every IC time course (continuous data = 1 trial; epoched data are concatenated with 1100 NaNs between epochs):

1. **Beat detection** — `heart_peak_detect(signal, srate, cfg_peak)`: 1–100 Hz FIR filter → squared z-score, peaks > 10 (i.e. |z| > 3.16) at ≥ 0.35 s apart → average 1 s template beat → polarity flip if the signal's skewness is negative → slide template over the signal → normalized "correlation" `cr` → beats = peaks of `cr` > `corthresh` (SASICA sets 0.2; heart_functions default is 0.6) → Q/S/P/T located around each R.
2. **Five measures per IC** (stored in `meas`):
   - `sk`, `ku` — skewness and kurtosis of the template-correlation trace `cr` (a spiky, peaky `cr` means a repeating QRS-like shape).
   - `RR` — CV (std/mean) of R-R intervals after keeping only the 0–70th percentile (drops long intervals from missed beats).
   - `Rampl` — CV of |R amplitude| after keeping the 15–85th percentile.
   - `bpm` — beats ÷ non-NaN duration in minutes.
   - `RPeakstoNoise` — median |R| ÷ median |signal between beats (±50 ms from R)|. **Computed and reported, not used in the decision.**
3. **Decision** — `NotCardiac(ic,1:5)` flags; `icarejCARACAS(ic) = ~any(NotCardiac(ic,:))`. Zero, one, or several ICs can pass. No ranking, no "best IC" — if the project needs exactly one, choosing among passers is a method decision for the user.

Full algorithm, every parameter and default, and output fields: `references/caracas_sasica.md`.

## 3. Running it — what must be true

Minimal working call (continuous data, CARACAS only):

```matlab
cfg = SASICA('getdefs');
for f = fieldnames(cfg)', if isstruct(cfg.(f{1})) && isfield(cfg.(f{1}),'enable'), cfg.(f{1}).enable = false; end, end
cfg.CARACAS.enable = true;
cfg.opts.noplot = 1; cfg.opts.noplotselectcomps = 1;   % otherwise it opens GUIs and uiwait()s
EEG = eeg_SASICA(EEG, cfg);
is_cardiac = EEG.reject.SASICA.icarejCARACAS;           % logical 1 x ncomp
meas       = EEG.reject.SASICA.icaCARACAS;              % struct array; meas(1).cfg = settings used
```

Checklist (each item has bitten someone — details in `references/environment.md`):

- **EEG struct** needs `icawinv`, `icaweights`, `icasphere`, `icachansind`, `srate`, `data`, and **channel locations with X/Y/Z** — `eeg_SASICA` errors "No electrode locations provided" even though CARACAS never uses them. If `EEG.icaact` is filled, it is used as-is.
- **Sampling rate > 200 Hz**, or set `cfg.CARACAS.cfg_peak.lpfreq` below Nyquist — `heart_peak_detect` hard-errors when its 100 Hz low-pass exceeds fs/2.
- **Paths**: SASICA root, `SASICA/CARACAS/heart_functions` (git submodule — empty unless cloned with `--recurse-submodules`), FieldTrip (`ft_defaults`; `ft_preprocessing` does the filtering), and EEGLAB functions (`eeg_emptyset`, `eeg_getdatact`, `convertlocs`) — SASICA ships a minimal copy in `SASICA/eeglab`.
- **This Windows machine has no full MATLAB, no FieldTrip and an empty heart_functions submodule** — see `references/environment.md`; ask the user before any MATLAB step.
- **No Statistics Toolbox on the Mac setup** (R2026a + Signal Processing only). `kurtosis`/`skewness`/`nanmean`/`range` come from FieldTrip `external/stats` (auto-added by `ft_defaults`); `prctile` is in base MATLAB R2026a (`toolbox/matlab/datafun/prctile.m`; older releases needed the Statistics Toolbox); **`zscore` is nowhere** — without a public `zscore.m` on the path, `heart_peak_detect` crashes in `nanzscore`. Use the shims in `hr/matlab/` (also shadows `prctile` — verified numerically identical to the built-in, max diff 1e-16). SASICA's `private/zscore.m` is invisible to heart_functions because of MATLAB's private-folder scoping.
- **`ft_SASICA` is broken as shipped**: `ft_SASICA.m:70` calls `comp2eeglab(comp,cfg,data)` but the function is declared `comp2eeglab(cfg,comp,data)`. Build the EEG struct yourself (as `hr/caracas/caracas_session.m` does) or pass the args in the declared order via a local fix you tell the user about.
- **Headless**: keep `noplot`/`noplotselectcomps` = 1 and all `cfg_peak.plot*` = 0, or MATLAB blocks on `ginput`/`questdlg`.
- **Runtime**: the template correlation is a per-sample loop, cost ∝ duration × fs² (template is fs+1 samples). Measured here: 10 min at 250 Hz = 0.6 s per IC; extrapolates to roughly a minute per IC for 1 h at 1000 Hz. The first `ft_preprocessing` call in a fresh MATLAB costs 10–50 s of warm-up. Don't "optimize" by changing the algorithm.
- **Edge beats**: SASICA's 1100-NaN padding spreads through the FIR filters by (hp order + lp order)/2 samples — measured 224 extra NaN samples (~0.9 s) at 250 Hz — so the last ~1 s of each epoch/recording yields no beats.

## 4. Getting a cardiac IC to exist (ICA side)

CARACAS can only find a heart IC that ICA separated. When a session yields no passer, the project's stance is: treat it as a decomposition problem (e.g. more components), not a lost session. First diagnose *why* by looking at the five `meas` values for the best-looking IC against each threshold — which criterion failed tells you whether the problem is separation (low sk/ku, high Rampl CV: cardiac mixed with other sources), detection (bpm off by ×2 / ×0.5: T-waves counted or beats missed), or physiology (true HR outside 34–97).

ICA background, what a cardiac IC looks like, and the levers that change separation (component count, high-pass, rank, data length, reference) — with which ones are sourced and which are only common practice: `references/eeg_ica_cardiac.md`.

## 5. How to answer

- Cite the file and line for any claim about behavior (`eeg_SASICA.m:777`, `heart_peak_detect.m:285`), and the commit date when a version matters.
- When the user's question touches a threshold or setting, give the current default, where it's set, and whether it's a SASICA default (sourced) or a local choice.
- If a step needs a method the toolkits don't define (picking one IC among several passers, beat cleaning, HRV), don't choose — follow the project's "How to propose a method" format.
- Never add or tighten criteria that drop ICs/beats/sessions on your own; CARACAS's five criteria are the published method, anything beyond is a rejection gate the user must request.
