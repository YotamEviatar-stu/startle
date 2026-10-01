# EEG, ICA and the cardiac component — domain reference

Use this to reason about *why* a cardiac IC does or doesn't appear. Items are tagged: **[toolkit]** = what the local code does/recommends, **[lit]** = standard published practice (cite before adopting), **[heuristic]** = common rule of thumb, not a validated method. Per the project rules, any change to the pipeline based on [lit]/[heuristic] items goes to the user as a proposal.

## Where the heart signal comes from

- The heart's electric field reaches every scalp electrode. The EEG-recorded "ECG artifact" is a few µV (vs. ~1 mV on a chest lead), largest between electrodes far apart along the heart-dipole axis, so low/lateral rim sites tend to carry more than the vertex. Check the actual topography rather than assuming channels. [lit, general EEG artifact literature]
- Its shape follows the QRS complex: a sharp spike (R) with Q/S deflections, a broader T-wave ~0.2–0.4 s later. Polarity at a channel depends on reference and position — that's why `heart_peak_detect` flips polarity by skewness. [toolkit]
- Distinct from the **pulse artifact** (electrode over an artery moving/impedance changing): smoother, delayed ~0.2–0.3 s after R, focal on one or few electrodes. An IC can capture pulse rather than ECG; its "R" timing would then lag the true R. [lit]

## What ICA gives you

- ICA unmixes channels into maximally independent sources: each IC has a **topography** (column of `icawinv` / `comp.topo`) and a **time course** (row of `icaact` / `comp.trial`). CARACAS only looks at time courses. [toolkit]
- A good cardiac IC: time course looks like an ECG (regular spikes at 0.6–1.7 s intervals), spectrum with harmonics at the heart rate, topography a broad smooth gradient (often left-right or diagonal) rather than a focal spot. ICLabel's "Heart" class encodes the same features (Pion-Tonachini et al. 2019, NeuroImage 198:181–197; implementation `mne-icalabel` / EEGLAB ICLabel — SASICA's minimal EEGLAB bundles the plugin). [lit]
- The heart can end up **split** over two ICs (e.g. QRS in one, T-wave or pulse in another) or **mixed** into an IC with other activity (then Rampl CV and sk/ku degrade). [lit/heuristic]

## Levers that change whether a cardiac IC separates

| Lever | What the local code does | Notes |
|---|---|---|
| Number of ICs | CARACAS README example: `pop_runica(...,'extended',1,'pca',round(dataRank/5))`; `hr/caracas/caracas_session.m` uses `round((nchan−1)/5)` | [toolkit] With 256 channels that is ~51 ICs. Weak sources like the ECG can be lost in the PCA reduction if too few are kept; more components is the project's first-line response to "no cardiac IC". |
| Rank | Average reference removes 1 rank → `rank = nchan − 1` in the wrapper | [lit] Interpolated channels also remove rank. Running ICA with more components than rank gives duplicated/complex ICs → SASICA errors "ICA decomposition seems bad". |
| High-pass before ICA | wrapper: 1 Hz firws on each channel block | [lit] 1–2 Hz high-pass improves ICA decompositions (Winkler et al. 2015, EMBC; Klug & Gramann 2021, EJN). heart_peak_detect independently band-passes 1–100 Hz. |
| Algorithm | extended infomax (`runica`, `extended = 1`), `rng(123)` | [lit] Lee, Girolami & Sejnowski 1999. Seed matters: ICA solutions vary run to run; fix the seed for reproducibility. |
| Amount of data | continuous whole session | [heuristic] EEGLAB guidance: ≥ k·n² samples with k ≈ 20–30 (Onton & Makeig 2006). 51 ICs → ≥ ~52 000–78 000 samples; long sessions satisfy it easily. |
| Reference | average | [lit] Changes each channel's cardiac projection but not whether ICA can isolate it. |
| Sampling rate | native | [toolkit] Must stay > 200 Hz for heart_peak_detect's 100 Hz low-pass unless `lpfreq` is lowered. |

## Reading CARACAS output diagnostically

For the IC that *should* be cardiac (look at its time course), compare each measure with its threshold:

| Failing criterion | Most likely meaning |
|---|---|
| bpm ≈ 2× true HR | T-waves (or pulse) counted as beats — template correlation ≥ 0.2 at T; check `corthresh` vs detection |
| bpm ≈ ½ true HR / RR CV high | missed beats (weak or mixed IC), long intervals — the 0–70 pct RR trim only partly absorbs this |
| bpm outside 34–97 but beats look right | physiology outside the prior (tachy-/bradycardia) |
| sk / ku low | template match isn't peaky: cardiac mixed with other sources, or IC isn't cardiac |
| Rampl CV high | amplitude drifts across the session (nonstationary mixing, movement, electrode change) |

Then decide (with the user) whether the fix is on the decomposition side (more ICs, different segment) — not by loosening thresholds silently.

## Reference list (add PDFs + `papers/INDEX.md` lines before adopting any of these)

- Chaumon, Bishop & Busch 2015, J Neurosci Methods 250:47–63 — SASICA; measures guide the experimenter's decision. (in `papers/`)
- Pion-Tonachini, Kreutz-Delgado & Makeig 2019, NeuroImage 198:181–197 — ICLabel.
- Winkler, Debener, Müller & Tangermann 2015, IEEE EMBC — high-pass filtering and ICA.
- Klug & Gramann 2021, Eur J Neurosci 54:8406–8420 — ICA preprocessing and data amount.
- Lee, Girolami & Sejnowski 1999, Neural Computation 11:417–441 — extended infomax.
- Onton & Makeig 2006, Prog Brain Res 159:99–120 — ICA data-length rule of thumb.
