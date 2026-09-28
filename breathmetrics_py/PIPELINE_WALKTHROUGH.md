# Stats pipeline: what the code actually does

Rewritten 2026-09-28 for `final_pipeline.py`, which now produces every table on its own.
Sample rate throughout is 1000 Hz, so 1 sample = 1 ms. Step numbers match the `# Step N`
comments in `final_pipeline.py`.

## 0. The chain

```
airflow_cache.pkl (trigger times, rest spans)  ─┐
ratings CSV (code, has_sound, label)           ─┼─► Layer 1 (slow, cached)  steps 1-5
raw .mff  (Airflow channel, signal2.bin)       ─┘       one BreathMetrics run per session
                                                         └─► _cache/final_pipeline_breaths.pkl
                                                 Layer 2 (fast, every run)  steps 6-14
                                                         └─► _out_final/*.csv
```

Layer 1 is rebuilt only when a setting in `FRONT_END` / `BREATHMETRICS` changes (the cache stores
them and compares on load) or when `FORCE_PREPROCESSING = True` in `main`. Layer 2 reads
`pipeline_config.py` on every run, so exclusions, rest spans and manual verdicts take effect
without rebuilding Layer 1.

Outputs in `_out_final/`: `breaths.csv`, `trial_verdicts.csv`, `rest_reference.csv`,
`trials_final.csv`, `subject_medians.csv`, `session_qc.csv`, `skipped.csv`.

---

## 1. Session selection and trigger check

**`build_breath_cache`** · `final_pipeline.py:166`
Lists the subject folders under `RAW_DATA_DIR`. For each subject × {eve, mor} it skips the pair if it
is in `SUBJECTS_EXCLUDE`, if its ratings CSV or MFF is missing, or if it is not in the trigger
cache; every skip is written to `skipped.csv` with its reason. If an MFF fails alignment, it tries
the next candidate MFF.

**`session_trials`** · `final_pipeline.py:72` (Step 1)
From `airflow_cache.pkl` (dated 2026-08-23; the code that built it is not in the repo) each trial
gets its picture-code time, D110 (sound) time and code value, and the session gets its D102
pre- and post-rest spans. A trial with no D105 or no code time is dropped.

**`validate_against_csv`** · `final_pipeline.py:85` (Step 1b)
Checks each trial against the ratings CSV:
- the trial count must match the CSV row count;
- each trial's code value must equal the CSV `trigger_num` at the same row;
- a D110 must be present exactly when the CSV says `has_sound`.

Any mismatch skips the session. Trigger **times** are not re-checked, only values and presence.
The CSV's `subjective_label` (1 = Negative, 2 = Neutral) and `has_sound` are then attached to
each trial (`process_session`, `final_pipeline.py:151`).

## 2. Raw signal → flow

**`load_mff`** · `load_mff.py:106` (Step 2)
Takes the `.mff` folder and channel `"Airflow"`. It reads only the PNS file (`signal2.bin`),
block by block, and multiplies by 1.0, or by 1e-6 if the unit is µV. It returns the raw 1000 Hz
pressure trace with no resampling, and caches it as `.npz` in `_cache/`.

**`epoch_clock`** · `load_mff.py:58`
Reads `epochs.xml` (recording pauses). It returns a function that maps "sample index / 1000"
(time in the concatenated recording) to the recording's clock, so breath times line up with
trigger times.

**`lowpass`** · `pressure.py:7`
Off (`lowpass_hz = 0`), so it just returns a copy.

**`estimate_zero`** (method `histogram`) · `pressure.py:18`
Takes the raw pressure and slides a 60 s window in 30 s steps. In each window it builds a 200-bin
histogram over the 1st–99th percentile range and takes the centre of the fullest bin as "zero
pressure". It places each value at its window centre and linearly interpolates to every sample
(held flat before the first centre and after the last). Output: a slowly moving zero line.

**`pressure_to_flow`** · `pressure.py:60` (Step 3)
Subtracts the zero line, then applies `flow = sign(d)·√|d|`. The output is the square-root
"flow" signal.

## 3. BreathMetrics: cleaning the flow (Step 4)

**`fft_smooth`** · `fft_smooth.py:4`
A moving average of width *w*, applied twice (so the kernel is triangular), computed by FFT. It is
**circular**: the end of the recording wraps into the start. This is the same as MATLAB's
`fftSmooth`.

**`BreathMetrics.__init__`** · `breathmetrics.py:18`
Smooths the flow with `fft_smooth(w = 50 samples = 50 ms)`.

**`correct_to_baseline`** (method `sliding`) · `breathmetrics.py:68`
Takes the smoothed flow, removes one straight-line trend fitted to the whole recording, then
subtracts `fft_smooth(w = 60,000 samples = 60 s)` of that detrended trace. No z-scoring. The result
is the signal that every later step uses (called "resp" below).

## 4. BreathMetrics: finding each breath (Step 4)

**`find_respiratory_extrema`** · `extrema.py:4`
1. Sets thresholds from the whole session: a peak must be above `mean + SD/2` and a trough below
   `mean − SD/2`.
2. Chops resp into windows of 100, 300, 700, 1000 and 5000 ms, each at 3 start offsets (0, ½, ⅔ of
   a window). That is 15 passes.
3. In each window, the maximum gets a peak vote if it is above the threshold, and the minimum gets a
   trough vote if it is below its threshold.
4. Picks the vote cut-off automatically: the level where the count of surviving candidates drops
   least. It keeps samples whose votes reach that cut-off.
5. Enforces peak → trough → peak alternation. Of two peaks in a row it keeps the higher; of two
   troughs, the lower.

Output: peak (max inspiratory flow) and trough indices. `simplify` then trims both lists to equal
length.

**`find_respiratory_pauses_and_onsets`** · `onsets_pauses.py:41`
Uses `n_bins = 20`, the default at 1000 Hz.
- **Trough → next peak** (the lead-up to an inhale): builds a 20-bin amplitude histogram of the
  segment.
  - If the fullest bin is a middle bin (6–14) **and** holds at least 5× the average bin count,
    there is an **expiratory pause**. The pause band is that bin plus up to 2 neighbours holding
    more than 25% of its count. Pause onset = first sample in the band; **inhale onset** = last
    sample in the band + 2.
  - Otherwise, **inhale onset** = the last sample at or below the whole-recording mean, + 1 (the
    upward zero-crossing).
- **Peak → next trough**: the same logic finds an inspiratory pause and the **exhale onset**. Here
  the band can widen by up to **5** neighbouring bins, which is hard-coded in MATLAB too
  (`findRespiratoryPausesAndOnsets.m:182`).
- **Edge breaths**: the first inhale onset comes from a histogram of the stretch before the first
  peak. The last exhale onset is the first sample after the last peak that falls below the mean.

**`find_respiratory_offsets`** · `offsets.py:4`
- Inhale offset = 1 sample before the exhale onset, or before the inspiratory pause if there is one.
- Exhale offset = 1 sample before the next inhale onset, or before the expiratory pause.
- Last breath: exhale offset = the first positive sample after its exhale onset. It is kept only if
  the exhale length is between ¼ and 1.75× the mean exhale length; otherwise it is NaN.

**`find_breath_durations`** · `durations.py:4`
- **Ti** = (inhale offset − inhale onset) / 1000 s. **Te** = (exhale offset − exhale onset) / 1000 s.
  Both exclude the pauses.
- Pause durations are computed separately.

**`find_respiratory_volumes`** · `volumes.py:4`
**Inhale volume** = the sum of |resp| from inhale onset to inhale offset (inclusive) × 1000/1000
(the scale factor is 1 at 1000 Hz). It is an area in a.u.·ms under the sqrt-flow after smoothing
and baseline correction.

`get_secondary_features` (`features.py`) also runs here. It only produces a session summary dict
that nothing downstream reads.

## 5. Per-breath numbers

Written in **`breath_features`** · `final_pipeline.py:105` (Step 5).

| column | computed as |
|---|---|
| `onset_s` | inhale onset sample / 1000, mapped through `epoch_clock` |
| `ti_s` | from `find_breath_durations` |
| `ttot_s` | (next inhale onset − this onset) / 1000 on concatenated-sample time; NaN for the last breath |
| `vi_over_ti` | inhale volume / Ti (mean inspiratory flow proxy) |
| `mv_au_min` | 60 · inhale volume / Ttot (minute-ventilation proxy) |
| `manual_bad_span` | True if the breath **onset** falls in `[a, b)` of a `MANUAL_BAD_SPANS` entry (Step 8) |

There is **no automatic breath-level rejection**. The only per-breath exclusion is
`manual_bad_span`.

## 6. Breaths → trials

Every step below runs twice per session, once per **anchor** (**`anchor_trials`**,
`final_pipeline.py:234`, Step 6):
- `code`: the picture-code time, all trials;
- `sound`: the D110 time, sound trials only.

**`rest_spans`** · `final_pipeline.py:241` (Step 7)
The D102 spans from the trigger cache, replaced per session by `MANUAL_BASELINE_SPANS`
(`post: None` drops the post block).

**`analysed_span`** · `final_pipeline.py:260` (Step 9)
From the 3rd breath onset before the first anchor to the 3rd breath onset after the last anchor.
Breaths outside it cannot become response breaths.

**`assign_windows`** · `final_pipeline.py:271` (Step 10)
Labels every breath:
- `rest_pre` / `rest_post`: the onset is inside the rest span and the breath is not manual-bad.
- `hold`: the last breath whose onset is at or before the anchor time (the breath in progress when
  the picture or sound arrives). It is excluded.
- `response` 1, 2, 3: the next 3 breaths after the hold. The run stops early at the next trial's
  hold breath. A manual-bad breath or one outside the analysed span is **dropped, not replaced**
  by a 4th breath.
- `gap`: everything else.

Rest labels are written first, so a hold/response label overwrites a rest label where they meet.

**`trial_verdicts`** · `final_pipeline.py:297` (Step 11)
The only trial verdict, one row per trial in `trial_verdicts.csv` (trials with zero response
breaths included):
- fewer than 2 response breaths → `insufficient_response`;
- code value listed in `MANUAL_TRIAL_REJECT` → `manual`;
- a code value in `MANUAL_TRIAL_ACCEPT` clears both.

It never reads amplitude, volume or the condition.

## 7. Trials → change scores

**`rest_reference`** · `final_pipeline.py:375` (Step 12)
For each session and anchor, the rest value of each feature is the **median** over all rest
breaths, with pre and post **pooled** (for example AB22/eve: 25 breaths). Output: `*_fix`, plus
`*_fix_pre` / `*_fix_post` as diagnostics.

**`trial_change`** · `final_pipeline.py:390` (Step 13)
For each kept trial and each of the 4 features (`ti_s`, `ttot_s`, `vi_over_ti`, `mv_au_min`):
- response value `r` = the **mean** of that trial's 2–3 response breaths;
- rest value `b` = the session's rest median;
- output columns: `_delta = r − b`, `_pct = 100·(r/b − 1)`, `_log = ln r − ln b`.

Example, AB22/eve trial 0, Ti: `r = 2.258 s` (mean of 3 breaths), `b = 2.397 s`, so
`ti_s_pct = −5.8 %`.

**`subject_medians`** · `final_pipeline.py:416` (Step 14)
One row per anchor × subject × session × valence (Negative, Neutral, and All pooled): the median
of the trial `_pct` values and the trial count.

## Worth checking before deployment

1. All trigger times and rest spans come from `airflow_cache.pkl` (2026-08-23), whose builder code
   was removed. Only code values and D110 presence are re-checked against the CSV.
2. The paired Wilcoxon tests and the plots in `figures/` have no saved source; they were made from
   an earlier run's `trials_final.csv` / `subject_medians.csv`.
3. A D102 rest span can touch the task. In AB22/eve, `rest_post` starts at 743.725 s, and breath
   188 (onset 743.816 s) is the 3rd sound-anchored response breath of the last trial, so it counts
   as rest under the `code` anchor (25 rest breaths) and as response under `sound` (24).
4. In `trial_verdicts`, a code listed in both `MANUAL_TRIAL_ACCEPT` and `MANUAL_TRIAL_REJECT` gets
   reason `manual_accept` and is marked **rejected**. No code is in both lists today, so it has no
   effect on current outputs.
