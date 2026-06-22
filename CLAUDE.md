# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context

This is **academic research**. Results must be statistically defensible, not merely visually appealing. Every analytical choice — filtering cutoffs, baseline windows, epoch timing, artifact rejection thresholds — must be justifiable and consistent across signals and sessions.

**Central hypothesis:** Morning physiological reactivity is calmer than Evening. Sleep reduces autonomic and respiratory responses to startling stimuli. The analysis must demonstrate this with real effect sizes and statistical tests (Wilcoxon signed-rank preferred given small N), not just trends in plots.

**Active pipelines:** HR (`HR/`) and Airflow (`Airflow/`). Both must be analysed to converge on the same conclusion, strengthening the claim across independent physiological channels.

**Canonical reference:** EMG raw (`scripts/emg_raw_potentiation.py`) defines the event logic, trial structure, DIN trigger handling, and baseline approach. All pipelines must be consistent with it.

## Pipeline Conventions

When adapting or modifying a pipeline, read the EMG reference first and mirror its structure exactly — do not introduce custom approaches unless explicitly asked. For deeper cross-pipeline consistency checks, invoke `/cross-pipeline-audit`.

Key technical dimensions that must be tuned rigorously and consistently:
- **Filtering** — bandpass / highpass / lowpass cutoffs appropriate to the signal (HR vs. Airflow have different frequency content); document the choice.
- **Baseline** — pre-stimulus baseline window length and reference method (mean subtraction, z-score) must match across conditions and sessions.
- **Epoch timing** — onset offset relative to DIN trigger, epoch length, and any pre/post padding must be principled and matched to the EMG reference.
- **Artifact handling** — flag or exclude trials with implausible values (e.g. HR outside 40–180 BPM); do not silently average over bad data.
- **Aggregation** — report per-trial values and condition means; do not collapse across conditions unless explicitly asked.

## Skills

Three project skills are available and should be used proactively:

- `/startle-experiment` — full briefing on experiment design, data layout, CSV structure, DIN triggers, and EMG as canonical reference. Invoke at the start of any new session before touching HR or Airflow code.
- `/cross-pipeline-audit` — audits HR and/or Airflow code against EMG raw as the canonical reference. Invoke whenever HR or Airflow code is written or reviewed.
- `/tune-pipeline` — autonomous parameter tuner for HR and Airflow pipelines. Runs coordinate-descent over analysis parameters, writes the best config to disk, and prints a full change log.

## Plotting

- Show RAW, segmented/cropped signal without high-pass filtering unless explicitly asked.
- Never use naive stride-based downsampling — use decimation that preserves peaks.
- Avoid over-zoomed Y-axes and limit sample counts so MNE does not freeze the machine.
- When the user asks for a "timecourse", plot the per-trial value over time — NOT an epoch-averaged or mean value across trials.
- Every plot must serve a statistical or diagnostic purpose; decorative plots are not the goal.

## Communication Style

Keep explanations concise and concrete. Do not raise theoretical code-review findings (e.g. hardcoded sample rate) when the data is uniform — don't invent issues that don't apply to this dataset. When reporting results, always include the test statistic and p-value alongside any visual summary.

## Data & Caching

Recordings are uniform; assume a consistent sample rate and don't add defensive handling for non-uniform rates unless told otherwise. Cache only downsampled traces, never full 1000 Hz traces, to avoid multi-minute load times.

## Data Layout

```
RAW_DATA_DIR/
  <SubjectID>/
    EEG/         ← contains *_eve_*.mff and *_mor_*.mff files
    startle output/   ← or CSVs directly in subject folder
      *_1.csv    ← evening ratings
      *_2.csv    ← morning ratings
```

**Event alignment:** DIN trigger channels (prefix `D`) are detected by amplitude threshold. Trigger 101 = session start, 124 = session end, 110 = startle. Events are matched to CSV rows by sequential order (first startle event → first CSV row).

**EMG script** (`scripts/emg_raw_potentiation.py`): original, standalone startle EMG analysis. Same data directory conventions and EGI MFF format, single-file script with inline config. This is the canonical processing reference for event logic, baseline, and trial structure.
