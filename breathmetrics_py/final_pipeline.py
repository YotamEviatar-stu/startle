import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breathmetrics_py import pipeline_config as cfg
from extras.emg_raw_potentiation import (find_startle_output_folder, find_mff_candidates,
                                         find_csv_by_suffix)
from extras.trial_epochs import load_all_trials_ratings, TriggerAlignmentError

from breathmetrics_py.load_mff import load_mff, epoch_clock
from breathmetrics_py.pressure import pressure_to_flow
from breathmetrics_py.breathmetrics import BreathMetrics

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = cfg.RAW_DATA_DIR
LAYER1_CACHE = cfg.TRIGGER_CACHE
CACHE_DIR = os.path.join(HERE, "_cache")
BREATH_CACHE = os.path.join(CACHE_DIR, "final_pipeline_breaths.pkl")
OUTPUT_DIR = os.path.join(HERE, "_out_final")

# Signal and BreathMetrics settings
CHANNEL = "Airflow"
SRATE = 1000.0

FRONT_END = {"transform": "sqrt", "zero_method": "histogram", "zero_win_sec": 60.0,
             "zero_nbins": 200, "zero_percentile": 50.0,
             "lowpass_hz": 0.0, "lowpass_order": 4}
BREATHMETRICS = {"smooth_win_ms": 50.0, "baseline_method": "sliding", "z_score": False,
                 "simplify": True, "n_bins": None}

# Trial-window settings
N_RESPONSE_MAX = 3
N_RESPONSE_MIN = 2
N_EDGE_BREATHS = 3

# Labels and output columns
ANCHORS = ("code", "sound")
REST = ("rest_pre", "rest_post")
FEATURES = ("ti_s", "ttot_s", "vi_over_ti", "mv_au_min")
LABEL_NAMES = {1: "Negative", 2: "Neutral"}

BREATH_COLUMNS = ["subject", "session", "anchor", "breath", "window", "trial_index",
                  "cycle_pos", "code_value", "code_time_s", "anchor_time_s", "label",
                  "label_name", "has_sound", "onset_s", "inhale_offset_s",
                  "exhale_onset_s", "exhale_trough_s", "exhale_offset_s",
                  "ti_s", "te_s", "ttot_s", "ti_over_ttot", "inhale_pause_s",
                  "exhale_pause_s", "inhale_volume_au", "exhale_volume_au",
                  "peak_insp_flow_au", "vi_over_ti", "mv_au_min", "manual_bad_span",
                  "trial_rejected", "trial_reject_reason"]
TRIAL_META = ["subject", "session", "anchor", "trial_index", "code_value", "code_time_s",
              "anchor_time_s", "label", "label_name", "has_sound"]


def cache_params():
    return {"channel": CHANNEL, "srate": SRATE, **FRONT_END, **BREATHMETRICS}


def excluded(subj, sess):
    for key in (subj, f"{subj}/{sess}"):
        if key in cfg.SUBJECTS_EXCLUDE:
            return key
    return None


# Step 1: read each trial's trigger times and the rest spans from airflow_cache.pkl
def session_trials(layer1, subj, sess):
    e = layer1[subj][sess]
    trials = [{"trial_index": int(t["trial_index"]),
               "code_value": int(t["code_value"]),
               "code_time": float(t["code_time"]),
               "d110_time": None if t.get("d110_time") is None else float(t["d110_time"])}
              for t in e["trials_meta"]
              if t.get("d105_time") is not None and t.get("code_time") is not None]
    rest = {"rest_pre": e.get("baseline_pre_sec"), "rest_post": e.get("baseline_post_sec")}
    return trials, rest


# Step 1b: check the pkl trials against the ratings CSV (count, picture code, sound)
def validate_against_csv(subj, sess, trials, ratings_df):
    where = f"{subj}/{sess}"
    if len(trials) != len(ratings_df):
        raise TriggerAlignmentError(
            f"{where}: layer1 has {len(trials)} trials, CSV has {len(ratings_df)} rows")
    for t in trials:
        row = ratings_df.iloc[t["trial_index"]]
        if t["code_value"] != int(row["trigger_num"]):
            raise TriggerAlignmentError(
                f"{where} trial {t['trial_index']}: layer1 code {t['code_value']} "
                f"!= CSV trigger_num {row['trigger_num']}")
        if (t["d110_time"] is not None) != bool(row["has_sound"]):
            raise TriggerAlignmentError(
                f"{where} trial {t['trial_index']}: D110 "
                f"{'present' if t['d110_time'] is not None else 'absent'} "
                f"but CSV has_sound={bool(row['has_sound'])}")


# ===== Layer 1 (slow, cached): one BreathMetrics run per session =====

def breath_features(mff_path):
    # Step 2: load the raw Airflow channel from the MFF
    signal, srate = load_mff(mff_path, CHANNEL, SRATE, cache_dir=CACHE_DIR)
    # Step 3: pressure -> flow (moving zero line + square root)
    flow, _ = pressure_to_flow(signal, srate, **FRONT_END)
    # Step 4: run BreathMetrics (smoothing, baseline, extrema, onsets, offsets, durations, volumes)
    bm = BreathMetrics(flow, srate, "humanAirflow", BREATHMETRICS["smooth_win_ms"])
    bm.estimate_all_features(baseline_method=BREATHMETRICS["baseline_method"],
                             z_score=BREATHMETRICS["z_score"],
                             simplify=BREATHMETRICS["simplify"],
                             n_bins=BREATHMETRICS["n_bins"])

    # Step 5: per-breath features, times mapped to the recording clock
    clock = epoch_clock(mff_path)
    onset_cat = np.asarray(bm.inhale_onsets, dtype=float) / srate
    n = onset_cat.size
    ttot = np.full(n, np.nan)
    ttot[:-1] = np.diff(onset_cat)
    ti = np.asarray(bm.inhale_durations, dtype=float)
    vol = np.asarray(bm.inhale_volumes, dtype=float)
    troughs = np.full(n, np.nan)
    tr = np.asarray(bm.exhale_troughs, dtype=float)[:n] / srate
    troughs[:tr.size] = tr

    return pd.DataFrame({
        "breath": np.arange(n),
        "onset_s": clock(onset_cat),
        "inhale_offset_s": clock(np.asarray(bm.inhale_offsets, dtype=float) / srate),
        "exhale_onset_s": clock(np.asarray(bm.exhale_onsets, dtype=float) / srate),
        "exhale_trough_s": clock(troughs),
        "exhale_offset_s": clock(np.asarray(bm.exhale_offsets, dtype=float) / srate),
        "ti_s": ti,
        "te_s": np.asarray(bm.exhale_durations, dtype=float),
        "ttot_s": ttot,
        "ti_over_ttot": ti / ttot,
        "inhale_pause_s": np.asarray(bm.inhale_pause_durations, dtype=float),
        "exhale_pause_s": np.asarray(bm.exhale_pause_durations, dtype=float),
        "inhale_volume_au": vol,
        "exhale_volume_au": np.asarray(bm.exhale_volumes, dtype=float),
        "peak_insp_flow_au": np.asarray(bm.peak_inspiratory_flows, dtype=float),
        "vi_over_ti": vol / ti,
        "mv_au_min": 60.0 * vol / ttot,
    })


# Steps 1-5 for one session
def process_session(subj, sess, mff_path, ratings_df, layer1):
    trials, rest = session_trials(layer1, subj, sess)
    validate_against_csv(subj, sess, trials, ratings_df)
    if "rest_pre" not in rest_spans(f"{subj}/{sess}", rest):
        raise TriggerAlignmentError(f"{subj}/{sess} has no baseline_pre_sec in the layer1 cache")
    # add the CSV label (Negative/Neutral) and has_sound to each trial
    for t in trials:
        row = ratings_df.iloc[t["trial_index"]]
        t["label"] = int(row["subjective_label"])
        t["has_sound"] = bool(row["has_sound"])
    return {"mff": mff_path, "trials": trials, "rest": rest,
            "breaths": breath_features(mff_path)}


# Build or load the breath cache for every subject x session
def build_breath_cache(layer1, subjects=None, force=False):
    # importing the breath cache (rebuilt if the settings above changed)
    cache = {"params": cache_params(), "sessions": {}}
    if os.path.exists(BREATH_CACHE) and not force:
        with open(BREATH_CACHE, "rb") as fh:
            stored = pickle.load(fh)
        if stored.get("params") == cache["params"]:
            cache = stored
        else:
            print("Breath cache built with other parameters, rebuilding.")

    if not os.path.isdir(RAW_DATA_DIR):
        raise FileNotFoundError(f"RAW_DATA_DIR not mounted: {RAW_DATA_DIR}")
    subject_folders = sorted(d for d in os.listdir(RAW_DATA_DIR)
                             if os.path.isdir(os.path.join(RAW_DATA_DIR, d)))
    if subjects:
        subject_folders = [s for s in subject_folders if s in subjects]

    # loop over subjects and sessions; process only those not cached yet
    skipped, changed = [], False
    for subj in subject_folders:
        subj_path = os.path.join(RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        startle_folder = find_startle_output_folder(subj_path)
        if not startle_folder or not os.path.isdir(eeg_folder):
            skipped.append((subj, "-", "missing folders"))
            continue

        for sess, sess_info in cfg.SESSION_MAP.items():
            why = excluded(subj, sess)
            if why:
                skipped.append((subj, sess, f"SUBJECTS_EXCLUDE ({why})"))
                continue
            if (subj, sess) in cache["sessions"]:
                continue
            csv_path = find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            candidates = find_mff_candidates(eeg_folder, sess)
            if not csv_path or not candidates:
                skipped.append((subj, sess, "missing csv or mff"))
                continue
            if sess not in layer1.get(subj, {}):
                skipped.append((subj, sess, "not in layer1 cache"))
                continue
            ratings_df = load_all_trials_ratings(csv_path)

            for mff_path in candidates:
                try:
                    entry = process_session(subj, sess, mff_path, ratings_df, layer1)
                except TriggerAlignmentError as e:
                    skipped.append((subj, sess, f"alignment: {e}"))
                    continue
                cache["sessions"][(subj, sess)] = entry
                changed = True
                print(f"  {subj}/{sess}: {len(entry['breaths'])} breaths, "
                      f"{len(entry['trials'])} trials")
                break

    # save the breath cache
    if changed or force:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(BREATH_CACHE, "wb") as fh:
            pickle.dump(cache, fh)
    return cache, skipped


# ===== Layer 2 (fast, re-run every time): windows, verdicts, change scores =====

# Step 6: pick the anchor: picture code (all trials) or D110 sound (sound trials only)
def anchor_trials(trials, anchor):
    if anchor == "code":
        return [dict(t, anchor_time=t["code_time"]) for t in trials]
    return [dict(t, anchor_time=t["d110_time"]) for t in trials if t["d110_time"] is not None]


# Step 7: rest spans (from the pkl, overridden by MANUAL_BASELINE_SPANS)
def rest_spans(key, rest):
    spans = dict(rest)
    manual = getattr(cfg, "MANUAL_BASELINE_SPANS", {}).get(key, {})
    if "pre" in manual:
        spans["rest_pre"] = manual["pre"]
    if "post" in manual:
        spans["rest_post"] = manual["post"]
    return {name: tuple(map(float, sp)) for name, sp in spans.items() if sp is not None}


# Step 8: flag breaths whose onset falls in a MANUAL_BAD_SPANS entry
def manual_bad_span(key, onset):
    bad = np.zeros(onset.size, dtype=bool)
    for a, b in getattr(cfg, "MANUAL_BAD_SPANS", {}).get(key, []):
        bad[(onset >= float(a)) & (onset < float(b))] = True
    return bad


# Step 9: analysed span: 3 breaths before the first anchor to 3 after the last
def analysed_span(onset, anchor_times):
    if onset.size == 0 or not anchor_times:
        return None
    first_i = int(np.searchsorted(onset, min(anchor_times), side="left"))
    last_i = int(np.searchsorted(onset, max(anchor_times), side="left"))
    lo_i = max(0, first_i - N_EDGE_BREATHS)
    hi_i = min(onset.size - 1, last_i + N_EDGE_BREATHS - 1)
    return float(onset[lo_i]), float(onset[hi_i])


# Step 10: label every breath: rest_pre / rest_post / hold / response 1-3 / gap
def assign_windows(onset, trials, spans, bad, span):
    n = onset.size
    window = np.array(["gap"] * n, dtype=object)
    tidx = np.full(n, -1, dtype=int)
    pos = np.zeros(n, dtype=int)
    for name, (lo, hi) in spans.items():
        window[(onset >= lo) & (onset < hi) & ~bad] = name
    skip = bad.copy()
    if span is not None:
        skip |= (onset < span[0]) | (onset > span[1])

    ordered = sorted(trials, key=lambda t: t["anchor_time"])
    hold = [int(np.searchsorted(onset, t["anchor_time"], side="right")) - 1 for t in ordered]
    for h in hold:
        if h >= 0:
            window[h] = "hold"
    for j, t in enumerate(ordered):
        stop = hold[j + 1] if j + 1 < len(hold) else n
        for p, i in enumerate(range(hold[j] + 1, min(hold[j] + 1 + N_RESPONSE_MAX, stop)),
                              start=1):
            if not skip[i]:
                window[i], tidx[i], pos[i] = "response", t["trial_index"], p
    return window, tidx, pos


# Step 11: trial verdict: < 2 response breaths or manual reject; manual accept overrides
def trial_verdicts(key, trials, window, tidx):
    reject_codes = set(getattr(cfg, "MANUAL_TRIAL_REJECT", {}).get(key, ()))
    accept_codes = set(getattr(cfg, "MANUAL_TRIAL_ACCEPT", {}).get(key, ()))
    verdicts = {}
    for t in trials:
        k = t["trial_index"]
        n_resp = int(np.sum((tidx == k) & (window == "response")))
        manual = t["code_value"] in reject_codes
        if t["code_value"] in accept_codes:
            reason = "manual_accept" if manual else ""
        else:
            reason = "+".join(r for r, hit in (("insufficient_response", n_resp < N_RESPONSE_MIN),
                                               ("manual", manual)) if hit)
        verdicts[k] = {"n_resp": n_resp, "rejected": bool(reason), "reason": reason}
    return verdicts


# Steps 6-11 for one session and one anchor
def session_anchor_rows(subj, sess, entry, anchor):
    key = f"{subj}/{sess}"
    b = entry["breaths"]
    onset = b["onset_s"].to_numpy(float)
    trials = anchor_trials(entry["trials"], anchor)
    spans = rest_spans(key, entry["rest"])
    bad = manual_bad_span(key, onset)
    span = analysed_span(onset, [t["anchor_time"] for t in trials])
    window, tidx, pos = assign_windows(onset, trials, spans, bad, span)
    verdicts = trial_verdicts(key, trials, window, tidx)

    by_k = {t["trial_index"]: t for t in trials}
    meta = {c: np.full(onset.size, np.nan) for c in
            ("code_value", "code_time_s", "anchor_time_s", "label", "has_sound")}
    rejected = np.zeros(onset.size, dtype=bool)
    reason = np.full(onset.size, "", dtype=object)
    for k, t in by_k.items():
        hit = tidx == k
        meta["code_value"][hit] = t["code_value"]
        meta["code_time_s"][hit] = t["code_time"]
        meta["anchor_time_s"][hit] = t["anchor_time"]
        meta["label"][hit] = t["label"]
        meta["has_sound"][hit] = t["has_sound"]
        rejected[hit] = verdicts[k]["rejected"]
        reason[hit] = verdicts[k]["reason"]

    rows = b.assign(subject=subj, session=sess, anchor=anchor, window=window,
                    trial_index=tidx, cycle_pos=pos, **meta,
                    label_name=pd.Series(meta["label"]).map(LABEL_NAMES).to_numpy(),
                    manual_bad_span=bad, trial_rejected=rejected,
                    trial_reject_reason=reason)[BREATH_COLUMNS]

    verdict_rows = pd.DataFrame([{
        "subject": subj, "session": sess, "anchor": anchor,
        "trial_index": k, "code_value": t["code_value"], "code_time_s": t["code_time"],
        "anchor_time_s": t["anchor_time"], "label": t["label"],
        "label_name": LABEL_NAMES[t["label"]], "has_sound": t["has_sound"],
        **verdicts[k]} for k, t in by_k.items()])

    qc = {"subject": subj, "session": sess, "anchor": anchor, "mff": entry["mff"],
          "n_breaths": int(onset.size), "n_trials": len(trials),
          "n_rejected": int(sum(v["rejected"] for v in verdicts.values())),
          "n_rest_pre": int(np.sum(window == "rest_pre")),
          "n_rest_post": int(np.sum(window == "rest_post")),
          "rest_pre": spans.get("rest_pre"), "rest_post": spans.get("rest_post"),
          "analysed_span": span}
    return rows, verdict_rows, qc


def nan_median(x):
    x = np.asarray(x, float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else np.nan


def nan_mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


# Step 12: rest reference: median of the pooled rest_pre + rest_post breaths per session
def rest_reference(breaths):
    keys = ["subject", "session", "anchor"]
    rows = []
    for key, g in breaths[breaths["window"].isin(REST)].groupby(keys):
        pre, post = g[g["window"] == "rest_pre"], g[g["window"] == "rest_post"]
        row = dict(zip(keys, key), n_rest=len(g), n_rest_pre=len(pre), n_rest_post=len(post))
        for c in FEATURES:
            row[c + "_fix"] = nan_median(g[c])
            row[c + "_fix_pre"] = nan_median(pre[c])
            row[c + "_fix_post"] = nan_median(post[c])
        rows.append(row)
    return pd.DataFrame(rows)


# Step 13: trial change: mean of the trial's response breaths vs the rest reference
def trial_change(breaths, verdicts, reference):
    keys = ["subject", "session", "anchor", "trial_index"]
    kept = verdicts[~verdicts["rejected"]]
    resp = breaths[breaths["window"] == "response"].groupby(keys)
    ref = reference.set_index(["subject", "session", "anchor"])

    rows = []
    for _, v in kept.iterrows():
        key = tuple(v[k] for k in keys)
        if key not in resp.groups:
            continue
        g = resp.get_group(key)
        row = {c: v[c] for c in TRIAL_META}
        row["n_resp"] = len(g)
        for c in FEATURES:
            r = nan_mean(g[c])
            b = ref.loc[key[:3], c + "_fix"] if key[:3] in ref.index else np.nan
            row[c + "_resp"], row[c + "_fix"] = r, b
            row[c + "_delta"] = r - b
            row[c + "_pct"] = 100.0 * (r / b - 1.0) if b > 0 else np.nan
            row[c + "_log"] = np.log(r) - np.log(b) if (b > 0 and r > 0) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# Step 14: per-subject median of trial pct, per valence and all trials pooled
def subject_medians(trials):
    pct = [c + "_pct" for c in FEATURES]
    keys = ["anchor", "subject", "session"]
    by_label = trials.groupby(keys + ["label_name"])
    pooled = trials.groupby(keys)
    out = pd.concat([
        by_label[pct].median().join(by_label.size().rename("n_trials")).reset_index(),
        pooled[pct].median().join(pooled.size().rename("n_trials")).reset_index()
        .assign(label_name="All"),
    ], ignore_index=True)
    return out[keys + ["label_name", "n_trials"] + pct].sort_values(
        keys + ["label_name"]).reset_index(drop=True)


# Steps 6-14 for every session and both anchors
def apply_analysis(cache):
    breaths, verdicts, qc = [], [], []
    for (subj, sess), entry in sorted(cache["sessions"].items()):
        if excluded(subj, sess):
            continue
        for anchor in ANCHORS:
            b, v, q = session_anchor_rows(subj, sess, entry, anchor)
            breaths.append(b)
            verdicts.append(v)
            qc.append(q)
    breaths = pd.concat(breaths, ignore_index=True)
    verdicts = pd.concat(verdicts, ignore_index=True)
    reference = rest_reference(breaths)
    trials = trial_change(breaths, verdicts, reference)
    return {"breaths": breaths, "trial_verdicts": verdicts, "rest_reference": reference,
            "trials_final": trials, "subject_medians": subject_medians(trials),
            "session_qc": pd.DataFrame(qc)}


def main(subjects=None):
    FORCE_PREPROCESSING = False
    SAVE_BREATH_TABLE = True

    # importing the trigger cache (airflow_cache.pkl)
    with open(LAYER1_CACHE, "rb") as fh:
        layer1 = pickle.load(fh)["sessions"]
    # Layer 1: breaths from the raw MFFs (cached)
    cache, skipped = build_breath_cache(layer1, subjects, FORCE_PREPROCESSING)
    if subjects:
        cache = dict(cache, sessions={k: v for k, v in cache["sessions"].items()
                                      if k[0] in subjects})
    if not cache["sessions"]:
        raise RuntimeError("no sessions produced breaths")

    # Layer 2: windows, verdicts, change scores
    out = apply_analysis(cache)
    out["skipped"] = pd.DataFrame(skipped, columns=["subject", "session", "reason"])

    # save outputs to OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name, df in out.items():
        if name == "breaths" and not SAVE_BREATH_TABLE:
            continue
        df.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)

    # print the trial-verdict summary
    v = out["trial_verdicts"]
    print("\n== Trial verdicts ==")
    for anchor, g in v.groupby("anchor"):
        print(f"  {anchor}: {len(g)} trials, {int(g['rejected'].sum())} rejected, "
              f"{len(out['trials_final'].query('anchor == @anchor'))} kept")
        print("    " + g.loc[g["rejected"], "reason"].value_counts().to_string()
              .replace("\n", "\n    "))
    print(f"-> {OUTPUT_DIR}  ({out['session_qc'][['subject', 'session']].drop_duplicates().shape[0]}"
          f" sessions, {len(skipped)} skipped)")
    return out


if __name__ == "__main__":
    main(subjects=sys.argv[1:] or None)
