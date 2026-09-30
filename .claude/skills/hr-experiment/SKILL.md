---
name: hr-experiment
description: Startle-experiment briefing for the HR repo — design, raw data layout, CSV columns, DIN trigger scheme, rest-block timeline, known bad files and subject quirks. Invoke before touching MFF loading, trial epochs, or rest/task spans for heart-rate work.
---

# Startle experiment — what the HR pipeline needs to know

## Design

Overnight sleep study. Each subject has two sessions: **Evening** (`eve`, CSV `_1`) before sleep and **Morning** (`mor`, CSV `_2`) after. In each session they view Negative / Neutral images; on some trials a loud startle probe (D110) plays mid-image. Afterwards they rate valence (1–9, low = negative) and arousal (1–9, high = aroused).

Session, image type and `has_sound` are **reporting metadata only**. The pipeline must never read them.

## Raw data layout

```
/Volumes/My Passport/startle_raw/
  <SubjectID>/
    EEG/*_eve_*.mff, *_mor_*.mff    (ignore *_SLEEP_*.mff)
    startle output/*_1.csv, *_2.csv
  subjects.xlsx                     ID, STAI-T
```

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

A fixed offset before the first D105 is **not** rest; it falls in the setup gap. Where the D102 pairs are missing or bad, `hr/config.py:MANUAL_REST_SPANS` holds hand-verified replacements (seconds; `None` = unusable).

## Known data issues

| session | issue |
|---|---|
| ES29 eve | first MFF unreadable; real one is `ES29_eve_2_*.mff` (D124 before the D102 pair) |
| MG14 mor | corrupt MFF (XML parse error) |
| ML28 eve | macOS `._` file picked up instead of the MFF |
| LO21 | folder contains MH20's MFF files |
| RP06 / SH25 | files named `RP6_*` / `ST25_*`; resolve by folder, not filename (`config.SUBJECT_ID`) |

These are data issues: skip them with try/except, don't hard-code around them.

## Colors

```python
NEG_COLOR = "#C0392B"; NEU_COLOR = "#2980B9"; REJ_COLOR = "#95A5A6"
EVE_COLOR = "#E67E22"; MOR_COLOR = "#8E44AD"
```
