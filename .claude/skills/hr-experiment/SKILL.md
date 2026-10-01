---
name: hr-experiment
description: Startle-experiment briefing for the HR repo — design, data layout (C:\startle_data 250 Hz FIFs), CSV columns, DIN trigger scheme, rest-block timeline, known bad files and subject quirks. Invoke before touching MFF loading, trial epochs, or rest/task spans for heart-rate work.
---

# Startle experiment — what the HR pipeline needs to know

## Design

Overnight sleep study. Each subject has two sessions: **Evening** (`eve`, CSV `_1`) before sleep and **Morning** (`mor`, CSV `_2`) after. In each session they view Negative / Neutral images; on some trials a loud startle probe (D110) plays mid-image. Afterwards they rate valence (1–9, low = negative) and arousal (1–9, high = aroused).

Session, image type and `has_sound` are **reporting metadata only**. The pipeline must never read them.

## Data layout

All data lives in `C:\startle_data` (`config.DATA_250_DIR`) — the only permanent storage:

```
C:\startle_data\
  <SubjectID>\
    <SubjectID>_eve_raw.fif, <SubjectID>_mor_raw.fif   250 Hz, built by hr/downsample.py
    startle output\*_1.csv, *_2.csv                   (+ .log, .psydat, *_demo = practice)
  downsample_log.csv                                  source MFF / failure per session
```

FIF contents: 257 EEG (`E1`–`E256`, `VREF`), `SpO2-Pulse`, all DIN stim channels. Exact trigger onsets (taken from the 1000 Hz MFF) are stored as **annotations** named by channel (`D102`, `D105`, …) — use those; the resampled stim channels can shift by a few ms. Not kept: `ECG` (mains noise in AB22 eve, not a real lead there), `Pleth`, EMG, other bio channels.

The raw MFFs were at `F:\startle_raw\<SubjectID>\EEG\*.mff` (`config.RAW_DATA_DIR`). That drive is temporary and will be removed — nothing may depend on it. `subjects.xlsx` (ID, STAI-T) was not found there.

CSV columns: `has_sound` (bool), `arousalRating`, `valenceRating`, `image_type` ("negative"/"neutral"), `trigger_num`, image name.

## DIN triggers

DIN channels are prefixed `D`. Threshold at 90% of channel max; a contiguous above-threshold run is one event (keep its first sample).

| code | meaning |
|---|---|
| D100 | practice block start (practice excluded) |
| D102 | rest-block marker, fires 4×: pre pair + post pair |
| D101 | task start |
| D105 | fixation onset, start of each trial |
| D{N} | picture identity; N == CSV `trigger_num` exactly |
| D110 | startle probe (sound trials only) |
| D124 | task end |
| D120–D123 | unrelated later phase, excluded |

Trial cycle: `D105 → D{code} → D110 (sound only) → next D105`. Match triggers to CSV rows by validating: the D105 count in [D101, D124] equals the CSV row count, and each trial's D{code} equals its `trigger_num`. Never match the Nth D110 to the Nth sound row.

## Session timeline

| span | boundaries | duration |
|---|---|---|
| rest pre | D102[0] → D102[1] | 60.22 s |
| setup gap | D102[1] → D101 | ~65 s, excluded |
| task | first picture code → D124 | varies |
| dead time | D124 → D102[2] | ~3 s, not rest |
| rest post | D102[2] → D102[3] | 60.21 s |

A fixed offset before the first D105 is **not** rest; it falls in the setup gap. Rest spans come only from the D102 markers and this timeline — there are no manual span overrides. How to handle sessions with missing or bad D102 pairs is a user decision at the alignment stage.

## Known data issues

| session | issue |
|---|---|
| ES29 eve | first MFF unreadable; real one is `ES29_eve_2_*.mff` (D124 before the D102 pair) |
| MG14 mor | corrupt MFF (XML parse error) |
| ML28 eve | macOS `._` file picked up instead of the MFF |
| LO21 | folder contains MH20's MFF files |
| RP06 / SH25 | files named `RP6_*` / `ST25_*`; resolve by folder, not filename (`config.SUBJECT_ID`) |
| DA01 eve | raw file named `DA01_task_*`; user confirmed it is eve (`config.SESSION_FILE_ALIASES`) |
| AK12 | flat raw folder (no `EEG\` or `startle output\`), eve file named `AK12_eve2_*`; CSV files have `(2)`/`(3)` duplicates and an `A12_*` mor file (`config.SESSION_FILE_ALIASES`) |
| OA02 mor | raw file named `OA_mor_*` |
| EL04 eve + mor | recorded at 250 Hz natively (no resampling) |

These are data issues: skip them with try/except, don't hard-code around them — except the aliases the user confirmed in `config.SESSION_FILE_ALIASES`.

## Colors

```python
NEG_COLOR = "#C0392B"; NEU_COLOR = "#2980B9"; REJ_COLOR = "#95A5A6"
EVE_COLOR = "#E67E22"; MOR_COLOR = "#8E44AD"
```
