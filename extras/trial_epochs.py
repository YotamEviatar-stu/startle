"""
Shared trial/epoch construction for all Startle pipelines (EMG, HR, Airflow).

Defines the validated per-trial cycle within the real task block (between D101
and D124):

    D105 (fixation/image onset) -> D{trigger_num} (picture identity, one-hot
    digital line whose channel number equals the CSV's trigger_num column
    exactly) -> D110 (startle probe, fires only on has_sound==True trials) ->
    next trial's D105.

epoch_id_n   = D105_n -> D105_{n+1}      (last epoch closed by D124 instead of
                                           a nonexistent D105_{n+1})
baseline_n   = [D105_n, code_n)           pure pre-stimulus fixation
response_n   = [code_n, code_{n+1})       picture display + subsequent fixation,
                                           ending when the next trial's picture
                                           code fires (last trial closed by D124,
                                           consistent with the epoch boundary rule)

This does NOT filter by has_sound -- every CSV row gets a trial. Use
load_all_trials_ratings() to load the CSV, not emg_raw_potentiation's
load_and_classify_ratings() (which drops has_sound==False rows).

Validation is inline, not a side script: build_trial_epochs() raises
TriggerAlignmentError if the D105 count doesn't match the CSV row count, or if
any trial's picture code doesn't match that row's trigger_num. Pipelines
should let this raise rather than catch-and-continue -- a mismatch means the
trial/CSV alignment for this session cannot be trusted.
"""

import re
import pandas as pd

import extras.emg_raw_potentiation as emg

TRIGGER_SESSION_START = emg.TRIGGER_SESSION_START   # 101
TRIGGER_SESSION_END   = emg.TRIGGER_SESSION_END     # 124
TRIGGER_FIXATION      = 105
TRIGGER_STARTLE       = emg.TRIGGER_STARTLE         # 110

# Ground truth, measured directly from each session's own EGI event log
# (Events_255 DINs.xml -- discrete per-event timestamps, independent of the
# amplitude-threshold detection get_events_from_eeg does on the raw channel
# data), not assumed: D110 fires strictly AFTER its own trial's picture-
# identity code, consistently 2.1-4.1s later. Checked across 128 sound trials
# in 4 real sessions (AB22 eve/mor, AH19 eve/mor): gap ranges were
# [2.142,4.028] / [2.134,4.030] / [2.139,4.027] / [2.108,4.059] seconds --
# zero trials with D110 before code, zero outside this band. 4.5s gives
# margin above the observed max while staying far tighter than a
# [D105_n, D105_{n+1}) segment scan (response windows run ~13-20s, wide
# enough to silently pick up a D110 that belongs to neither this trial nor
# is a real match -- see the git history of this constant for that bug).
# Re-measure with inspect_triggers.py before changing this value; do not
# guess it.
D110_WINDOW_SEC = 4.5

_STRUCTURAL_RE = re.compile(r"^D(\d{3})$")   # D101, D105, D110, D124, ...
_CODE_DIN_RE   = re.compile(r"^DIN(\d)$")    # DIN1..DIN9   (single-digit codes)
_CODE_DI_RE    = re.compile(r"^DI(\d{2,})$") # DI10..DI99+  (two-digit+ codes)


class TriggerAlignmentError(Exception):
    """D105/code-channel counts or values don't match the session's CSV."""


def load_all_trials_ratings(csv_path: str) -> pd.DataFrame:
    """
    Load a startle ratings CSV with NO has_sound filtering -- every row is a
    real trial in this scheme (sound and no-sound alike). Adds the same
    subjective_label classification emg_raw_potentiation.load_and_classify_ratings
    uses (arousal >= 7 or valence <= 3 -> Negative), minus the has_sound filter.
    """
    df = pd.read_csv(csv_path).reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1  # Negative
        return 2  # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


def _classify_channel(channel: str):
    """Return ('structural', code) | ('picture', trigger_num) | (None, None)."""
    m = _STRUCTURAL_RE.match(channel)
    if m:
        return "structural", int(m.group(1))
    m = _CODE_DIN_RE.match(channel)
    if m:
        return "picture", int(m.group(1))
    m = _CODE_DI_RE.match(channel)
    if m:
        return "picture", int(m.group(1))
    return None, None


def _merge_contiguous(events_df: pd.DataFrame):
    """
    Collapse strictly-contiguous (gap == 1 sample) same-channel hits into one
    event, keeping the first sample -- matches the documented DIN convention.
    Empirically every event in this dataset is already 1 sample wide, but
    pipelines must not rely on that holding for every future recording.
    """
    merged = []
    for _, row in events_df.iterrows():
        ch, s = row["Channel"], int(row["Sample"])
        if merged and merged[-1]["Channel"] == ch and s - merged[-1]["LastSample"] == 1:
            merged[-1]["LastSample"] = s
        else:
            merged.append({"Channel": ch, "Sample": s, "LastSample": s})
    return merged


def first_session_marker_sample(raw):
    """The task block's D101 sample -- everything before this is pre-task
    idle time (impedance check, setup), not part of the recorded protocol."""
    events_df = emg.get_events_from_eeg(raw)
    merged = _merge_contiguous(events_df)
    channels = [m["Channel"] for m in merged]
    if "D101" not in channels:
        raise TriggerAlignmentError("D101 not found in this recording")
    return merged[channels.index("D101")]["Sample"]


def build_trial_epochs(raw, ratings_df: pd.DataFrame):
    """
    Build per-trial epoch boundaries for one session from a loaded (not yet
    cropped) MNE raw object and its full, unfiltered ratings DataFrame
    (see load_all_trials_ratings).

    Returns a list of dicts, one per CSV row, in CSV row order:
      {
        "trial_index"     : int,        0-based, == ratings_df row index
        "d105_sample"     : int,        fixation/image onset
        "code_sample"     : int,        picture-identity trigger fires here
        "code_value"      : int,        == ratings_df.iloc[i]["trigger_num"]
        "d110_sample"     : int | None, startle probe; None on no-sound trials
        "epoch_end_sample": int,        next trial's d105_sample, or the
                                         session's D124 sample for the last trial
        "baseline_range"  : (start, end)  = (d105_sample, code_sample)
        "response_range"  : (start, end)  = (code_sample, next trial's
                                         code_sample, or D124 for the last trial)
      }

    Raises TriggerAlignmentError if the D105 count doesn't equal len(ratings_df),
    if any trial doesn't have exactly one picture-code event, if a trial's code
    value doesn't match its CSV row's trigger_num, if more than one D110 falls
    within D110_WINDOW_SEC of a trial's code_sample, if a trial's D110
    presence/absence within that window doesn't match its CSV row's has_sound
    value, or if any D110 in the recording isn't within D110_WINDOW_SEC of any
    trial's code (an orphan probe).

    d110_sample is matched by proximity to code_sample (+/-D110_WINDOW_SEC),
    NOT by falling inside the [D105_n, D105_{n+1}) segment -- see
    D110_WINDOW_SEC's module-level comment. The window is symmetric (not
    code_sample .. code_sample+D110_WINDOW_SEC only) for robustness, but
    empirically, across every session measured so far, D110 always fires
    AFTER code_sample (2.1-4.1s later) and never before -- a negative
    d110_rel downstream (see airflow_glm.py) would be a genuine anomaly
    worth investigating, not the expected case.
    """
    if "trigger_num" not in ratings_df.columns:
        raise TriggerAlignmentError(
            "ratings_df has no 'trigger_num' column -- cannot validate picture codes"
        )

    events_df = emg.get_events_from_eeg(raw)
    merged = _merge_contiguous(events_df)
    channels = [m["Channel"] for m in merged]

    if "D101" not in channels or "D124" not in channels:
        raise TriggerAlignmentError("D101 or D124 not found in this recording")

    d101_i = channels.index("D101")
    d124_i = channels.index("D124", d101_i)
    window = merged[d101_i:d124_i + 1]
    d124_sample = window[-1]["Sample"]

    d105_events = [m for m in window if m["Channel"] == "D105"]
    n_expected = len(ratings_df)
    if len(d105_events) != n_expected:
        raise TriggerAlignmentError(
            f"D105 count ({len(d105_events)}) != CSV row count ({n_expected}). "
            f"Trial/CSV alignment for this session cannot be trusted."
        )

    d105_idxs_in_window = [i for i, m in enumerate(window) if m["Channel"] == "D105"]
    sfreq = float(raw.info["sfreq"])
    d110_window_samps = int(round(D110_WINDOW_SEC * sfreq))

    # All D110 events in the task block, found ONCE up front. D110 is matched
    # to a trial by proximity to that trial's own code_sample (+/-2s), not by
    # falling inside a [D105_n, D105_{n+1}) segment -- response windows run
    # ~13-20s, far wider than the 2s a real probe can ever be from its code,
    # so segment membership alone is not evidence a D110 belongs to that trial.
    all_d110_samples = [m["Sample"] for m in window if m["Channel"] == "D110"]

    trials = []
    for n, w_i in enumerate(d105_idxs_in_window):
        d105_sample = window[w_i]["Sample"]
        next_d105_sample = (
            window[d105_idxs_in_window[n + 1]]["Sample"]
            if n + 1 < len(d105_idxs_in_window)
            else d124_sample
        )
        segment = window[w_i + 1: (d105_idxs_in_window[n + 1] if n + 1 < len(d105_idxs_in_window) else len(window))]

        picture_hits = []
        for ev in segment:
            kind, value = _classify_channel(ev["Channel"])
            if kind == "picture":
                picture_hits.append((ev["Sample"], value))

        # Spurious single-sample blips on unrelated DI/DIN channels (crosstalk
        # or amplitude-threshold noise, not real picture-identity codes) do
        # occur alongside the genuine code within a trial window -- e.g. DI66,
        # DI73 firing once in an otherwise-clean session where trigger_num never
        # exceeds 40. Select by VALUE match against this row's expected
        # trigger_num rather than by count: if exactly one hit matches, use it
        # and ignore the rest as noise. Only raise when the match is genuinely
        # ambiguous (none match, or more than one matches -- which would mean
        # a real duplicate of the correct code, not just stray noise).
        expected_code = int(ratings_df.iloc[n]["trigger_num"])
        matching_hits = [h for h in picture_hits if h[1] == expected_code]

        if len(matching_hits) != 1:
            raise TriggerAlignmentError(
                f"Trial {n} (D105 @ sample {d105_sample}): expected CSV "
                f"trigger_num {expected_code}, found {len(matching_hits)} "
                f"matching picture-code event(s) among {len(picture_hits)} "
                f"total candidate(s) {[h[1] for h in picture_hits]}. Trial/CSV "
                f"alignment for this session cannot be trusted."
            )
        code_sample, code_value = matching_hits[0]

        # D110 has no identity value to cross-validate against (unlike the
        # picture code), so proximity to code_sample (+/-D110_WINDOW_SEC) is
        # the only available ground-truth check -- and cross-check the
        # resulting count against ratings_df's has_sound column (ground truth
        # for "was a probe played this trial"), the same way trigger_num is
        # cross-checked for the picture code above.
        d110_in_window = [s for s in all_d110_samples
                           if abs(s - code_sample) <= d110_window_samps]
        expected_has_sound = bool(ratings_df.iloc[n]["has_sound"])
        if len(d110_in_window) > 1 or (len(d110_in_window) >= 1) != expected_has_sound:
            raise TriggerAlignmentError(
                f"Trial {n} (D105 @ sample {d105_sample}, code @ {code_sample}): "
                f"CSV has_sound={expected_has_sound}, found {len(d110_in_window)} "
                f"D110 event(s) {d110_in_window} within {D110_WINDOW_SEC}s of the "
                f"code. Trial/CSV alignment for this session cannot be trusted."
            )
        d110_sample = d110_in_window[0] if d110_in_window else None

        trials.append({
            "trial_index": n,
            "d105_sample": d105_sample,
            "code_sample": code_sample,
            "code_value": code_value,
            "d110_sample": d110_sample,
            "epoch_end_sample": next_d105_sample,
            "baseline_range": (d105_sample, code_sample),
            "response_range": None,  # filled in below once all code_samples are known
        })

    # response_n = [code_n, code_{n+1}); last trial closed by D124, same rule as
    # the epoch boundary itself.
    for n, trial in enumerate(trials):
        next_code_sample = trials[n + 1]["code_sample"] if n + 1 < len(trials) else d124_sample
        trial["response_range"] = (trial["code_sample"], next_code_sample)

    # Completeness check: every D110 in the task block must have been claimed
    # by exactly one trial above. Consecutive codes are ~13-20s apart (response
    # window duration) vs a 4s-wide (+/-2s) match window per trial, so windows
    # never overlap -- a leftover, unclaimed D110 here means a probe fired
    # more than D110_WINDOW_SEC from every trial's code, contradicting the
    # ground-truth timing constraint, and must not be silently dropped.
    claimed = {t["d110_sample"] for t in trials if t["d110_sample"] is not None}
    orphans = [s for s in all_d110_samples if s not in claimed]
    if orphans:
        raise TriggerAlignmentError(
            f"{len(orphans)} D110 event(s) {orphans} are not within "
            f"{D110_WINDOW_SEC}s of any trial's code_sample. Trial/CSV "
            f"alignment for this session cannot be trusted."
        )

    return trials
