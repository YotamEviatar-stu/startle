# CARACAS inside SASICA (`eeg_SASICA.m`) — full reference

Source of truth: `SASICA/eeg_SASICA.m` lines 699–795 (CARACAS block), defaults in `SASICA/SASICA.m` `getdefs` (lines ~880–891), detector `SASICA/CARACAS/heart_functions/heart_peak_detect.m`. Versions: SASICA master `9ab76d7` (2026-05-22); heart_functions submodule pinned at `0db5c42` (2025-07-25). `heart_peak_detect.m` is byte-identical at heart_functions master `194e35d` (2026-05-04) — the 7 newer commits only add a Python port (`heart_py/`) and relicense to BSD-3.

## Contents
1. Entry points
2. Configuration (every field, default, meaning)
3. Algorithm step by step
4. heart_peak_detect in detail
5. Outputs
6. Threshold history
7. Gotchas and limitations

---

## 1. Entry points

| Entry | Signature | Notes |
|---|---|---|
| `eeg_SASICA` | `[EEG, cfg] = eeg_SASICA(EEG, cfg)` | The real engine. Command-line, scriptable. |
| `SASICA` | `SASICA` (GUI), `SASICA('getdefs')` (default cfg), `SASICA([], 'key', val)` | GUI stores prefs with `setpref('SASICA','cfg',...)` — a GUI session can change what "defaults" means for later GUI runs; `getdefs` always returns the coded defaults. |
| `ft_SASICA` | `[cfg] = ft_SASICA(cfg, comp, data)` | FieldTrip wrapper. **Broken**: line 70 passes `(comp,cfg,data)` to `comp2eeglab(cfg,comp,data)`. Also calls `rm_frompath eeglab` and adds SASICA's minimal EEGLAB copy. |
| EEGLAB menu | Tools → SASICA (`eegplugin_SASICA.m`, version string 'SASICA 1.4') | Interactive. |

`cfg` fields are merged with defaults by `setdef` (recursive), so pass only what you change — but note every non-CARACAS method's `enable` default is already `false`, and `eeg_SASICA` errors if *no* method is enabled.

## 2. Configuration — `cfg.CARACAS`

| Field | Default | Used as |
|---|---|---|
| `enable` | `false` | Must be `true`. |
| `thresh_sk` | `1.0` | NotCardiac(:,1) if `sk < thresh_sk` |
| `thresh_ku` | `5.3` | NotCardiac(:,2) if `ku < thresh_ku` |
| `thresh_RR` | `0.31` | NotCardiac(:,3) if `RR > thresh_RR` |
| `thresh_Rampl` | `0.23` | NotCardiac(:,4) if `Rampl > thresh_Rampl` |
| `thresh_bpm` | `[34 97]` | NotCardiac(:,5) if `bpm < 34 || bpm > 97` |
| `prctl_RR` | `[0 70]` | Keep RR intervals between these percentiles before CV |
| `prctl_Rampl` | `[15 85]` | Keep |R| amplitudes between these percentiles before CV |
| `cfg_peak` | `corthresh=0.2, absPT=0, abstemplate=0, NaNST=0` | Passed verbatim to `heart_peak_detect`; any heart_peak_detect option can be added here (e.g. `lpfreq`, `mindist`) |

`cfg.opts`: `noplot` (0), `noplotselectcomps` (0), `nocompute` (0), `legfig` (1), `FontSize` (14). For batch runs set `noplot=1, noplotselectcomps=1`; otherwise `pop_selectcomps` opens and `uiwait` blocks.

Thresholds were tuned by the SASICA author ("optimized", commits 2026-02 → 2026-05). The repo cites no paper or dataset for the optimization.

## 3. Algorithm (per IC `i_comp`)

```
icaacts = eeg_getdatact(EEG,'component',1:ncomp)   % uses EEG.icaact if non-empty
ECG_candidate = [icaacts(i,:,:) NaN(1,1100,ntrials)]; ECG_candidate = ECG_candidate(:)'
HeartBeats = heart_peak_detect(ECG_candidate, EEG.srate, cfg_peak)     % legacy vector call
```

- **NaN padding**: each epoch is followed by 1100 NaN samples (commit `fde5486`, 2026-02-04 "add NaNs between trials to avoid overlap"). For continuous data this is 1100 trailing NaNs.
- **RPeakstoNoise** = median(|x(R)|) / median(|x| over each inter-beat span trimmed 50 ms after R and 50 ms before next R). Reported only.
- **sk, ku** = `HeartBeats(1).sk/.ku`, computed on the template-correlation trace (§4).
- **RR** = `diff([HeartBeats.R_time])`, keep `prctile` 0–70, CV = std/mean.
- **Rampl** = `|x(R_sample)|`, keep `prctile` 15–85, CV = std/|mean|.
- **bpm** = `numel(HeartBeats) / (sum(~isnan(x))/srate/60)` — NaN padding excluded.
- **Decision**: `rej = true(1,ncomp); rej(any(NotCardiac,2)) = 0`. In SASICA's vocabulary "rej" = *selected*; for CARACAS selected = cardiac.

Every IC is scored independently — no z-scoring across ICs, no ranking. So the result does not depend on how many ICs there are, except through what ICA separated.

## 4. `heart_peak_detect` (heart_functions, Chaumon 2016–2025)

Call forms: `(ECG_vector, fs [, cfg])` legacy (SASICA uses this), `(cfg, data)` with FieldTrip raw data + `cfg.channel`, `(cfg)` reading `cfg.dataset`.

Defaults (`heart_peak_detect.m` ~line 258): `hpfilter='yes', hpfreq=1, hpfilttype='firws', lpfilter='yes', lpfreq=100, lpfilttype='firws', thresh=10, mindist=0.35, corthresh=0.6, PRmax=0.25, QRmax=0.05, RSmax=0.1, QTmax=0.42, FixSlarger=0, absPT=0, NaNST=0, abstemplate=0`, all plot options 0.

Steps:
1. `ft_preprocessing` band-pass 1–100 Hz (firws). **Errors if `lpfreq > fs/2`.** Requires exactly one trial.
2. `FixSlarger` (optional): NaN out positive (>0) or negative (<0) samples before step 3.
3. `ECG2z = nanzscore(x).^2`; `peakseek(ECG2z, thresh=10, mindist*fs)` → first-pass R candidates (only need enough for a template).
4. Template `mHB` = nanmean of ±0.5 s windows around candidates.
5. Polarity: if `sign(skewness(ECG)) == -1`, flip ECG and template. Note the local `skewness` (bottom of file) is **uncentered**: `mean(x.^3)/mean(x.^2)^1.5`.
6. Zero-pad 1000 samples each side (NaN→0), optional `abs` (`abstemplate`), sliding dot product with template (a plain loop over every sample), normalize by max → `cr`.
7. R peaks = `peakseek(cr, corthresh, mindist*fs)`. With SASICA's `corthresh=0.2`, any sample where the template match is ≥ 20 % of the best match can become a beat.
8. Q = min in [R−QRmax, R]; P = max in [R−PRmax, Q] (abs if `absPT`); S = min in [R, R+RSmax]; T = max in [S, Q+QTmax].
9. `NaNST`: set `cr` to NaN over S→T spans before moments.
10. `sk = skewness(cr)` (uncentered local version), `ku = kurtosis(cr)` (MATLAB/FieldTrip `kurtosis`, centered, non-excess: Gaussian = 3).

Implied limits: `mindist = 0.35 s` caps detectable rate at ~171 bpm; the 1 s template assumes beats are ≥ ~0.5 s apart for a clean template.

## 5. Outputs

- `EEG.reject.SASICA.icarejCARACAS` — logical 1×ncomp, true = cardiac.
- `EEG.reject.SASICA.icaCARACAS` — struct array 1×ncomp with fields `RPeakstoNoise, sk, ku, RR, Rampl, bpm, NotCardiac` (1×5 logical in the order sk, ku, RR, Rampl, bpm); `icaCARACAS(1).cfg` holds the `cfg.CARACAS` used.
- `EEG.reject.SASICA.icarejCARACAScol` — plot colour.
- `EEG.reject.gcompreject` — OR across all enabled methods (CARACAS-only run: same as `icarejCARACAS`).
- `EEG.reject.SASICA.var` — variance of each IC activation.
- **Not returned**: the beat times. To get R peaks for the chosen IC, call `heart_peak_detect` again on that IC with the same `cfg_peak` (and same NaN padding if you want identical results).

## 6. Threshold history (`git log -p SASICA.m`)

| Commit | Date | sk | ku | RR | Rampl | bpm | other |
|---|---|---|---|---|---|---|---|
| `d6b8e2e` before | ≤2026-02-04 | 1.1 | 5.1 | — | 0.26 | [35 100] | RPeakstoNoise 10; absPT/abstemplate/NaNST = 1 |
| `d6b8e2e` | 2026-02-04 | 1.3 | 5.0 | — | 0.28 | [35 98] | RPeakstoNoise 3.3; absPT/abstemplate/NaNST → 0 |
| `26d1069` | 2026-02-05 | 1.4 | 5.1 | — | 0.27 | [34 97] | RPeakstoNoise threshold removed |
| `c2c7d7e` | 2026-02-11 | 1.2 | 5.0 | — | — | — | |
| `1aade2a` | 2026-05-21 | **1.0** | **5.3** | 0.31 | **0.23** | [34 97] | "final optimized CARACAS settings" |

(— = unchanged in that commit.) If results were produced before 2026-05-21, they used different thresholds; check `meas(1).cfg`.

## 7. Gotchas and limitations

- **Channel locations required** (`eeg_SASICA.m` ~line 150: `if isempty([EEG.chanlocs.X]) error`). CARACAS doesn't use them.
- **Complex ICA** → error "The ICA decomposition seems bad".
- **`zscore` without Statistics Toolbox** — needs a public shim on the path; see `environment.md`.
- **Filtering across NaN padding** — `heart_peak_detect` filters the NaN-padded vector through `ft_preprocessing`; SASICA silences `FieldTrip:dataContainsNaN`. Measured: NaNs spread by (hp order + lp order)/2 samples — 224 samples (~0.9 s) at 250 Hz (hp order 414, lp order 34) — so no beats in the last ~1 s of each epoch/recording.
- **ICs with < 2 detected beats** give NaN RR/Rampl → the NaN comparisons are false → those criteria don't flag; bpm and sk/ku still do.
- **Multiple passers** are common when the heart splits over 2 ICs (e.g. QRS vs. T-wave or pulse-related), or none pass when HR is outside [34, 97] (exercise, bradycardic athletes) — bpm range is a physiological prior, not measured on this cohort.
- **Epoched input**: SASICA pads 1100 NaNs per epoch; short epochs (< ~2 s) give very few beats per epoch and a poor template. Continuous data is the intended use for cardiac detection.
- **Runtime** scales with samples × template length (template = fs + 1 samples), i.e. duration × fs². Measured on this machine: 10 min of 250 Hz data = 0.6 s per IC. Downsampling (≥ 250 Hz so the 100 Hz low-pass stays valid) is the usual lever — but it's a preprocessing choice for the user.
- **Sanity result**: on a synthetic 70-bpm ECG (2 min, 250 Hz, noise SD 0.05), `heart_peak_detect` with SASICA's `cfg_peak` found 139/139 beats, sk 3.63, ku 18.85, bpm 69.5 — well above the sk/ku thresholds. Useful as a smoke test after any path change.
