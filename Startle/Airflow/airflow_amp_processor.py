# airflow_demo_processor.py — stateless processing for the Airflow pipeline (Demo/Testing).
#
# ── Cache architecture ─────────────────────────────────────────────────────────
#
# Layer 1  process_session()       MFF -> bandpass-filtered raw signal
#                                  (downsampled) + trial metadata. Written to
#                                  pickle. The 0.01-70 Hz Butterworth bandpass
#                                  (AIRFLOW_HIGHPASS/AIRFLOW_LOWPASS) runs here,
#                                  at native rate, before resampling. No NK2.
#
# Layer 2  apply_analysis_params() Everything else:
#            - NK2 feature extraction (peaks/troughs/rate/amplitude) on the
#              already-filtered full session signal; NK2's own cleaning is
#              disabled (RSP_PEAK_METHOD_CLEANING="none") so it isn't a
#              second bandpass filter
#            - Epoch cutting at the ground-truth trial boundaries from trial_epochs.py
#            - Baseline, rejection, scoring
#          Returns (trials_data, traces). Never touches the MFF.
#
# What requires FORCE_RELOAD (Layer 1 changes):
#   AIRFLOW_CHANNEL, CACHE_SFREQ, AIRFLOW_HIGHPASS, AIRFLOW_LOWPASS
#
# What is free to change in the notebook (Layer 2):
#   RSP_CLEAN_METHOD / RSP_PEAK_METHOD_CLEANING, all rejection gates,
#   AIRFLOW_SCORE_MAX, USE_SUBJECTIVE_TRIAL_TYPE
#
# Epoch/baseline/response window timing is NOT a free parameter anymore. It is
# ground truth derived from the D105 -> D{trigger_num} -> D105 trigger cycle
# (see Startle/trial_epochs.py) -- there is no TMIN/TMAX offset to tune. This
# pipeline's own DIN-trigger handling (STI 014 / find_events) is retired;
# trial_epochs.build_trial_epochs() is now the sole source of trial boundaries
# for every pipeline, EMG included.
#
# ── Signal notes ───────────────────────────────────────────────────────────────
#
# Bandpass: single Butterworth 0.01-70 Hz, 2nd order (AIRFLOW_HIGHPASS/
# AIRFLOW_LOWPASS in config), applied once in process_session() on the
# native-rate raw signal, before resampling to CACHE_SFREQ. This is a WIDE
# anti-alias / DC-removal pass, NOT respiration isolation -- NK2's own
# khodadad2018 cleaning (RSP_PEAK_METHOD_CLEANING) does the ~0.05-3 Hz
# respiration-band isolation the peak detector needs. Do NOT set cleaning to
# "none": the detector then runs on the wideband trace and finds ~1 breath/min.
# Respiratory content: 0.1-0.5 Hz at rest (12-30 bpm).

import os
import pickle
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.signal import butter, filtfilt

import trial_epochs

def _butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def _butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = _butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def process_session(mff_path, ratings_df, config):
    """Layer 1: ground-truth trial epochs (trial_epochs.build_trial_epochs) +
    raw signal extraction, downsampled for caching.

    ratings_df must be the FULL, unfiltered CSV (trial_epochs.load_all_trials_ratings)
    -- every row becomes a trial here, sound and no-sound alike.

    Event/epoch extraction always runs on the original full-rate, full-channel
    raw (read_raw_egi) -- DIN triggers must not be resampled or have other
    channels dropped before detection. Only after build_trial_epochs() succeeds
    do we pick the analysis channel and downsample for the cache.

    Raises trial_epochs.TriggerAlignmentError if this session's D105/D{code}
    triggers don't match its CSV -- callers must let this propagate (or
    explicitly catch and skip the session) rather than silently continuing
    with an untrustworthy trial/CSV alignment.
    """
    import mne
    if config.VERBOSE:
        print(f"  Layer 1: Processing raw MFF ({os.path.basename(mff_path)}) ...")

    try:
        raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  Error reading {mff_path}: {e}")
        return None

    native_sfreq = raw.info["sfreq"]
    trials = trial_epochs.build_trial_epochs(raw, ratings_df)  # raises on mismatch

    ch_names = raw.info["ch_names"]
    target_ch = next((c for c in ch_names if config.AIRFLOW_CHANNEL.lower() in c.lower()), None)
    if target_ch is None:
        if config.VERBOSE:
            print(f"  Channel '{config.AIRFLOW_CHANNEL}' not found in {ch_names}")
        return None

    raw.pick([target_ch])

    # ── Bandpass filter (native rate, before downsampling) ─────────────────────
    # Must run here: CACHE_SFREQ's Nyquist (12.5 Hz at 25 Hz) is below
    # AIRFLOW_LOWPASS (70 Hz), so this cutoff can't be realized after resample.
    raw.apply_function(
        lambda x: _butter_bandpass_filter(
            x.astype(np.float64), config.AIRFLOW_HIGHPASS, config.AIRFLOW_LOWPASS,
            native_sfreq, order=2),
        picks=0, channel_wise=True, verbose=False)

    raw.resample(sfreq=config.CACHE_SFREQ, npad="auto", verbose=False)

    cache_sfreq = raw.info["sfreq"]
    signal_raw  = raw.get_data(picks=0)[0]
    times_raw   = raw.times

    # Trigger sample indices are in native_sfreq space (from the un-resampled
    # raw); convert to seconds here so they're valid regardless of CACHE_SFREQ.
    trials_meta = []
    for t in trials:
        row = ratings_df.iloc[t["trial_index"]]
        trials_meta.append({
            "trial_index":        t["trial_index"],
            "code_value":         t["code_value"],
            "has_sound":          t["d110_sample"] is not None,
            "d105_time":          t["d105_sample"] / native_sfreq,
            "code_time":          t["code_sample"] / native_sfreq,
            "d110_time":          (t["d110_sample"] / native_sfreq
                                    if t["d110_sample"] is not None else None),
            "baseline_range_sec": tuple(s / native_sfreq for s in t["baseline_range"]),
            "response_range_sec": tuple(s / native_sfreq for s in t["response_range"]),
            "subjective_label":   row.get("subjective_label", np.nan),
            "image_type":         row.get("image_type", None),
            "image_detail":       row.get("file_name", None),
            "valence":            row.get("valenceRating", np.nan),
            "arousal":            row.get("arousalRating", np.nan),
        })

    payload = {
        "sfreq":       cache_sfreq,
        "channel":     target_ch,
        "signal_raw":  signal_raw,
        "times_raw":   times_raw,
        "trials_meta": trials_meta,
    }
    return payload


def apply_analysis_params(sessions_cache, config):
    """Layer 2: Run filters, extract respiratory features, score and clean trials.

    Accepts the same sessions_cache format as airflow_processor.apply_analysis_params:
      sessions_cache = {subj: {sess_key: {raw_signal, sfreq, startle_samps, trials_meta}}}

    Returns (trials_data, traces) — same nested format as the main processor — so
    airflow_raw_show.ipynb can swap this processor in by changing one import line.

    Extra capabilities vs the main processor:
      • Shape-centroid rejection gate (AIRFLOW_SHAPE_Z_THRESHOLD in config)
      • Post-stimulus flat-line rejection gate (AIRFLOW_POST_MIN_STD_RATIO in config)
    """
    trials_data = {}
    traces      = {}
    subjects_exclude = getattr(config, "SUBJECTS_EXCLUDE", {})

    for subj, sess_dict in sorted(sessions_cache.items()):
        if subj in subjects_exclude:
            continue
        trials_data[subj] = {}
        traces[subj]      = {}

        for sess_key, sess in sess_dict.items():
            raw_sig       = sess["signal_raw"]
            sfreq         = float(sess["sfreq"])
            trials_meta   = sess["trials_meta"]

            # ── NK2 on full session signal ────────────────────────────────────
            # raw_sig is the WIDE Layer-1 bandpass (0.01-70 Hz, native rate --
            # see process_session). NK2's khodadad2018 cleaning
            # (RSP_PEAK_METHOD_CLEANING) then isolates the respiration band so
            # the peak detector works -- it must NOT be "none" or detection
            # collapses to ~1 breath/min and every trial fails the BxB cycle test.
            try:
                rsp_signals, _ = nk.rsp_process(
                    raw_sig.astype(float),
                    sampling_rate=int(round(sfreq)),
                    method=config.RSP_CLEAN_METHOD,
                    method_cleaning=config.RSP_PEAK_METHOD_CLEANING)
            except Exception as exc:
                print(f"  NK2 failed for {subj} {sess_key}: {exc} — skipping")
                trials_data[subj][sess_key] = []
                continue

            clean    = rsp_signals["RSP_Clean"].to_numpy().astype(np.float32)
            nan_col  = np.full(len(clean), np.nan, dtype=np.float32)
            rsp_amp  = (rsp_signals["RSP_Amplitude"].to_numpy().astype(np.float32)
                        if "RSP_Amplitude" in rsp_signals.columns else nan_col.copy())
            rsp_rate = (rsp_signals["RSP_Rate"].to_numpy().astype(np.float32)
                        if "RSP_Rate" in rsp_signals.columns else nan_col.copy())
            zeros    = np.zeros(len(clean))
            peaks    = np.flatnonzero(rsp_signals.get("RSP_Peaks",   pd.Series(zeros)).to_numpy())
            troughs  = np.flatnonzero(rsp_signals.get("RSP_Troughs", pd.Series(zeros)).to_numpy())

            d110_secs = np.array([m["d110_time"] for m in trials_meta
                                   if m["d110_time"] is not None])
            traces[subj][sess_key] = {
                "signal":            clean,
                "sfreq":             sfreq,
                "startle_samps_sec": d110_secs,
            }

            # ── Cut one ground-truth epoch per trial ──────────────────────────
            # Span: [d105_time, response_range_sec[1]) -- baseline + response,
            # contiguous by construction (response_range starts exactly where
            # baseline_range ends, at code_time). t=0 is set at code_time (the
            # picture-identity trigger), so the baseline portion is tw<0 and
            # the response portion is tw>=0. Epoch length varies per trial
            # (no more shared WIDE_TMIN/TMAX) -- this is ground truth, not a
            # tunable window.
            trials = []
            for meta in trials_meta:
                d105_idx = int(round(meta["d105_time"] * sfreq))
                code_idx = int(round(meta["code_time"]  * sfreq))
                end_idx  = int(round(meta["response_range_sec"][1] * sfreq))
                if d105_idx < 0 or end_idx > len(clean) or end_idx <= d105_idx:
                    continue

                times_wide = ((np.arange(d105_idx, end_idx) - code_idx)
                              / sfreq).astype(np.float32)

                ep_peaks   = np.array([p - d105_idx for p in peaks
                                        if d105_idx <= p < end_idx], dtype=np.int64)
                ep_troughs = np.array([t_ - d105_idx for t_ in troughs
                                        if d105_idx <= t_ < end_idx], dtype=np.int64)

                trial = {
                    "epoch_raw_wide":       raw_sig[d105_idx:end_idx].copy(),
                    "epoch_clean_wide":     clean[d105_idx:end_idx].copy(),
                    "rsp_amplitude_wide":   rsp_amp[d105_idx:end_idx].copy(),
                    "rsp_rate_wide":        rsp_rate[d105_idx:end_idx].copy(),
                    "ep_peaks":             ep_peaks,
                    "ep_troughs":           ep_troughs,
                    "times_wide":           times_wide,
                    "sfreq":                sfreq,
                    "channel":              sess["channel"],
                    "trial_index":          meta["trial_index"],
                    "code_value":           meta["code_value"],
                    "has_sound":            meta["has_sound"],
                    "subjective_label":     meta["subjective_label"],
                    "label":                meta["subjective_label"],
                    "valence":              meta["valence"],
                    "arousal":              meta["arousal"],
                    "image_detail":         meta["image_detail"],
                    "image_type":           meta.get("image_type", ""),
                    "baseline_mean":        None,
                    "baseline_amp":         None,
                    "baseline_rate":        None,
                    "baseline_peak_time":   None,
                    "response_amp":         None,
                    "response_rate":        None,
                    "response_trough_time": None,
                    "response_peak_time":   None,
                    "score":                None,
                    "score_ventilation":    None,
                    "rejected":             None,
                    "rejection_reason":     "",
                    "epoch_anal":           None,
                    "times_anal":           None,
                }
                trials.append(trial)

            _run_demo_analysis(trials, config)
            trials_data[subj][sess_key] = trials

    return trials_data, traces


def _run_demo_analysis(trials, config):
    """BxB scoring + structural shape-centroid outlier rejection.

    Processing order:
      Pass 1 — label resolution, BxB cycle extraction, session-level statistics
      Shape  — centroid distance matrix (see detailed block below)
      Pass 2 — all rejection gates, BxB scoring, epoch trim, summary log
    """
    if not trials:
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1  Labels · BxB cycle extraction · session-level statistics
    # ══════════════════════════════════════════════════════════════════════════
    bl_stds       = []
    pre_masks     = []
    amp_baselines = []
    trial_bxb     = []

    for t in trials:
        # ── Condition label ──────────────────────────────────────────────────
        if config.USE_SUBJECTIVE_TRIAL_TYPE:
            t["label"] = t["subjective_label"]
        else:
            img = str(t.get("image_type") or t.get("image_detail", "")).lower().strip()
            if "negative" in img or img == "1":
                t["label"] = 1
            elif "neutral" in img or img == "2":
                t["label"] = 2
            else:
                t["label"] = t["subjective_label"]

        tw         = t["times_wide"]
        ep         = t["epoch_clean_wide"]
        ep_amp     = t.get("rsp_amplitude_wide")
        ep_rate    = t.get("rsp_rate_wide")
        ep_peaks   = t.get("ep_peaks",   np.array([], dtype=np.int64))
        ep_troughs = t.get("ep_troughs", np.array([], dtype=np.int64))

        # ── Baseline window stats (used by noisy_baseline and flat_signal gates) ──
        # Ground truth: the entire pre-code-onset span IS the baseline (tw<0 by
        # construction -- t=0 is code_time, the epoch starts at d105_time). No
        # BASELINE_TMIN/TMAX cutoff anymore; using less than the full fixation
        # window would silently discard real baseline data.
        pre_mask = tw < 0
        pre_masks.append(pre_mask)

        bs = float(np.std(ep[pre_mask])) if np.any(pre_mask) else np.nan
        bl_stds.append(bs)
        t["baseline_std"] = bs

        # ── session_breath_amp: median typical breath amplitude across all trials ──
        if ep_amp is not None and np.any(pre_mask):
            vals  = ep_amp[pre_mask]
            valid = vals[np.isfinite(vals) & (vals > 0)]
            if len(valid) > 0:
                amp_baselines.append(float(np.median(valid)))

        # ── BxB cycle extraction ─────────────────────────────────────────────
        # Baseline : last NK2 peak at or before t=0
        # Response : first NK2 peak after the first trough that follows t=0
        #            (= first COMPLETE post-probe breath)
        bxb = {"baseline_amp": np.nan, "baseline_rate": np.nan,
               "response_amp": np.nan, "response_rate": np.nan,
               "baseline_peak_time":   np.nan,
               "response_trough_time": np.nan,
               "response_peak_time":   np.nan,
               "has_cycles":           False}

        if len(ep_peaks) > 0 and ep_amp is not None and ep_rate is not None:
            peak_times    = tw[ep_peaks]
            pre_peak_idxs = ep_peaks[peak_times <= 0]

            post_trough_idxs = (ep_troughs[tw[ep_troughs] > 0]
                                if len(ep_troughs) > 0
                                else np.array([], dtype=np.int64))

            if len(pre_peak_idxs) > 0 and len(post_trough_idxs) > 0:
                last_pre          = pre_peak_idxs[-1]
                first_post_trough = post_trough_idxs[0]
                trough_t          = float(tw[first_post_trough])
                post_trough_peaks = ep_peaks[peak_times > trough_t]
                if len(post_trough_peaks) > 0:
                    first_post = post_trough_peaks[0]
                    b_amp  = float(ep_amp[last_pre])
                    b_rate = float(ep_rate[last_pre])
                    r_amp  = float(ep_amp[first_post])
                    r_rate = float(ep_rate[first_post])
                    if all(np.isfinite(v) and v > 0 for v in (b_amp, b_rate, r_amp, r_rate)):
                        bxb = {
                            "baseline_amp":        b_amp,
                            "baseline_rate":       b_rate,
                            "response_amp":        r_amp,
                            "response_rate":       r_rate,
                            "baseline_peak_time":   float(tw[last_pre]),
                            "response_trough_time": trough_t,
                            "response_peak_time":   float(tw[first_post]),
                            "has_cycles":           True,
                        }
        trial_bxb.append(bxb)

    # Session-level baseline statistics (used by z-score and flat gates)
    sess_std_median    = float(np.nanmedian(bl_stds)) if bl_stds else np.nan
    sess_std_std       = float(np.nanstd(bl_stds))    if bl_stds else np.nan
    session_breath_amp = float(np.nanmedian(amp_baselines)) if amp_baselines else np.nan

    # ══════════════════════════════════════════════════════════════════════════
    # SHAPE-CENTROID OUTLIER DETECTION
    #
    # Principle: a trial whose epoch waveform deviates strongly from the
    # session's average breath shape is flagged as structurally atypical.
    # This catches post-stimulus flatlines, breath-holding, and gross
    # movement/disconnect artifacts that share nothing in common with the
    # rest of the session — artifacts that slip past baseline-only gates
    # because the baseline window itself may look normal.
    #
    # Config parameters (all read from the config object):
    #   AIRFLOW_SHAPE_REJECTION_ENABLE  bool   master switch (default True)
    #   AIRFLOW_SHAPE_SD_THRESHOLD      float  cutoff = mean(MSD) + k*std(MSD)
    #                                          (default 3.0 SDs above mean)
    #   AIRFLOW_SHAPE_WINDOW_MIN        float  window start, seconds re t=0
    #   AIRFLOW_SHAPE_WINDOW_MAX        float  window end,   seconds re t=0
    #
    # Epoch length now varies per trial (ground-truth D105->code->D105 spans,
    # no shared WIDE_TMIN/TMAX) -- times_wide is NOT identical across trials
    # anymore. Each trial's own time axis is still grid-aligned to the same
    # dt=1/sfreq with t=0 exactly at an integer sample (code_time), so any
    # trial whose epoch fully covers [shape_win_min, shape_win_max) yields
    # exactly the same column count. Trials that don't fully cover the window
    # (mainly the last trial of a session, closed early by D124) are excluded
    # from the shape-centroid matrix rather than padded/truncated -- silently
    # forcing them into the matrix would compare unequal time spans.
    # ══════════════════════════════════════════════════════════════════════════

    shape_enable    = getattr(config, "AIRFLOW_SHAPE_REJECTION_ENABLE", True)
    shape_threshold = getattr(config, "AIRFLOW_SHAPE_SD_THRESHOLD",     3.0)
    shape_win_min   = getattr(config, "AIRFLOW_SHAPE_WINDOW_MIN",
                              getattr(config, "ANAL_TMIN", -5.0))
    shape_win_max   = getattr(config, "AIRFLOW_SHAPE_WINDOW_MAX",
                              getattr(config, "ANAL_TMAX",  10.0))

    # Per-trial MSD score and session-level cutoff — initialised to nan so
    # Pass 2 can test for them cleanly even when the block is skipped.
    shape_msd     = [np.nan] * len(trials)
    shape_cutoff  = np.nan
    n_shape_skipped_coverage = 0

    if shape_enable and len(trials) > 1:

        # ── Phase 1: Matrix Alignment ────────────────────────────────────────
        # Extract the shape window from every trial's own NK2-cleaned epoch,
        # masked against its own times_wide. Only trials whose epoch fully
        # covers [shape_win_min, shape_win_max) contribute a row -- their
        # column count is guaranteed identical to each other (same dt, same
        # t=0 alignment) but trials with a shorter epoch are excluded rather
        # than forced into the matrix.
        sfreq_ref      = trials[0]["sfreq"]
        expected_n_cols = int(round((shape_win_max - shape_win_min) * sfreq_ref))
        shape_rows, shape_row_trial_idxs = [], []
        for i, t in enumerate(trials):
            tw_t  = t["times_wide"]
            mask_t = (tw_t >= shape_win_min) & (tw_t < shape_win_max)
            if int(np.sum(mask_t)) == expected_n_cols:
                shape_rows.append(t["epoch_clean_wide"][mask_t].astype(np.float64))
                shape_row_trial_idxs.append(i)
            else:
                n_shape_skipped_coverage += 1
        n_cols = expected_n_cols

        if n_cols > 0 and len(shape_rows) > 1:
            waveform_matrix = np.vstack(shape_rows)          # (n_covered_trials × n_cols)

            # ── Z-score normalise each row (amplitude-independent shape comparison) ──
            # Subtract each trial's mean and divide by its std so that a deep breath
            # and a shallow breath with identical timing patterns produce identical rows.
            # Flatlines (std = 0) → row stays as zeros → maximum distance from centroid,
            # so genuine flatlines are still caught.
            row_means = waveform_matrix.mean(axis=1, keepdims=True)
            row_stds  = waveform_matrix.std(axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0                  # avoid div-by-zero
            waveform_matrix = (waveform_matrix - row_means) / row_stds

            # ── Phase 2: Centroid Vector Generation ──────────────────────────
            # The centroid is the mean z-scored waveform — the representative
            # breath shape for this subject and session, amplitude-independent.
            centroid = np.mean(waveform_matrix, axis=0)    # (n_cols,)

            # ── Phase 3: Outlier Evaluation via Mean Squared Distance ─────────
            # Each trial's MSD measures its average squared point-wise
            # deviation from the centroid. A perfect match gives MSD = 0;
            # a flatline, breath-hold, or disconnected channel gives a
            # large MSD relative to the session distribution.
            msd_per_row = np.mean(
                (waveform_matrix - centroid) ** 2, axis=1  # (n_covered_trials,)
            )

            # Statistical cutoff boundary:
            #   cutoff = mean(MSD) + AIRFLOW_SHAPE_SD_THRESHOLD × std(MSD)
            # Trials above this line are structural outliers for this session.
            # Computed only over covered trials -- excluded (short-epoch) trials
            # don't influence the cutoff or the centroid.
            msd_mean     = float(np.nanmean(msd_per_row))
            msd_std      = float(np.nanstd(msd_per_row))
            shape_cutoff = msd_mean + shape_threshold * msd_std

            # Scatter back into full trial order; excluded trials stay nan.
            for row_i, trial_i in enumerate(shape_row_trial_idxs):
                shape_msd[trial_i] = float(msd_per_row[row_i])

            # Attach per-trial diagnostics for notebook inspection
            for t, msd in zip(trials, shape_msd):
                t["shape_msd"]    = msd
                t["shape_cutoff"] = float(shape_cutoff)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2  Rejection gates · BxB scoring · epoch trim
    # ══════════════════════════════════════════════════════════════════════════
    for t, pre_mask, bxb, msd in zip(trials, pre_masks, trial_bxb, shape_msd):
        tw  = t["times_wide"]
        ep  = t["epoch_clean_wide"]
        bs  = t["baseline_std"]

        baseline_amp  = bxb["baseline_amp"]
        baseline_rate = bxb["baseline_rate"]
        response_amp  = bxb["response_amp"]
        response_rate = bxb["response_rate"]
        has_cycles    = bxb["has_cycles"]

        t["baseline_mean"]        = baseline_amp
        t["baseline_amp"]         = baseline_amp
        t["baseline_rate"]        = baseline_rate
        t["response_amp"]         = response_amp
        t["response_rate"]        = response_rate
        t["baseline_peak_time"]   = bxb["baseline_peak_time"]
        t["response_trough_time"] = bxb["response_trough_time"]
        t["response_peak_time"]   = bxb["response_peak_time"]
        t["has_cycles"]           = has_cycles

        # Gate 0: BxB prerequisite — no usable breath cycles found
        no_cycles_reject = not has_cycles

        # Gate 1: Baseline noise z-score — actively drifting baseline
        z_threshold  = getattr(config, "AIRFLOW_Z_SCORE_THRESHOLD", None)
        z_reject     = (bool((bs - sess_std_median) / sess_std_std > z_threshold)
                        if z_threshold is not None
                           and not np.isnan(bs) and sess_std_std > 0
                        else False)

        # Gate 2: Flat signal — sensor detached or probe fully off
        min_ratio    = getattr(config, "AIRFLOW_MIN_STD_RATIO", None)
        flat_reject  = (bool(bs < min_ratio * sess_std_median)
                        if min_ratio is not None
                           and not np.isnan(bs) and not np.isnan(sess_std_median)
                        else False)

        # Gate 3: Physiological RSP rate ceiling — movement / cough artifact
        rate_thresh  = getattr(config, "RSP_RATE_ARTIFACT_THRESHOLD", None)
        rate_reject  = False
        if rate_thresh is not None:
            ep_rate_arr  = t.get("rsp_rate_wide")
            valid_rates  = (ep_rate_arr[np.isfinite(ep_rate_arr)]
                            if ep_rate_arr is not None else np.array([]))
            if len(valid_rates) > 0:
                rate_reject = bool(float(np.max(valid_rates)) > rate_thresh)

        # Gate 4: Post-stimulus flat line — response window has no variance
        # Ground truth: response window is tw>=0 (everything from code_time
        # onward, the full response_range), not a RESPONSE_TMIN/TMAX offset.
        post_ratio       = getattr(config, "AIRFLOW_POST_MIN_STD_RATIO", None)
        post_flat_reject = False
        if post_ratio is not None and not np.isnan(sess_std_median):
            post_mask = tw >= 0
            post_std  = float(np.std(ep[post_mask])) if np.any(post_mask) else np.nan
            if not np.isnan(post_std):
                post_flat_reject = bool(post_std < post_ratio * sess_std_median)

        # Gate 5: Shape-centroid — waveform is a structural outlier for this session
        # A trial is rejected when its MSD exceeds the session-level cutoff:
        #   MSD_trial > mean(MSD_session) + AIRFLOW_SHAPE_SD_THRESHOLD * std(MSD_session)
        shape_reject = (shape_enable
                        and not np.isnan(msd)
                        and not np.isnan(shape_cutoff)
                        and bool(msd > shape_cutoff))

        reasons = []
        if no_cycles_reject: reasons.append("no_cycles_found")
        if z_reject:         reasons.append("noisy_baseline")
        if flat_reject:      reasons.append("flat_signal")
        if rate_reject:      reasons.append("rate_artifact")
        if post_flat_reject: reasons.append("flat_response")
        if shape_reject:     reasons.append("atypical_shape")

        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""

        # Trim to analysis window
        anal_mask       = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
        t["epoch_anal"] = ep[anal_mask]
        t["times_anal"] = tw[anal_mask]

        # BxB scores
        if t["rejected"] or not config.PERFORM_SCORING:
            t["score"] = t["score_ventilation"] = np.nan
        elif not np.isnan(session_breath_amp) and session_breath_amp > 0 and has_cycles:
            t["score"]             = (response_amp - baseline_amp) / session_breath_amp
            t["score_ventilation"] = (response_amp * response_rate
                                      - baseline_amp * baseline_rate) / session_breath_amp
        else:
            t["score"] = t["score_ventilation"] = np.nan

        # Gate 6: Score ceiling (applied post-scoring)
        score_max = getattr(config, "AIRFLOW_SCORE_MAX", None)
        if (score_max is not None and not t["rejected"]
                and t["score"] is not None and not np.isnan(t["score"])
                and abs(t["score"]) > score_max):
            t["rejected"]          = True
            t["score"]             = t["score_ventilation"] = np.nan
            t["rejection_reason"]  = "score_ceiling"

    # ── Shape-centroid summary ────────────────────────────────────────────────
    if shape_enable:
        n_shape  = sum(1 for t in trials
                       if "atypical_shape" in (t.get("rejection_reason") or ""))
        n_total  = len(trials)
        cutoff_s = f"MSD cutoff={shape_cutoff:.3e}" if not np.isnan(shape_cutoff) else "gate skipped"
        skip_s   = (f"  {n_shape_skipped_coverage} trial(s) excluded from matrix "
                    f"(epoch shorter than shape window)"
                    if n_shape_skipped_coverage else "")
        print(f"  [shape-centroid] {n_shape}/{n_total} trials flagged as atypical_shape  "
              f"window=[{shape_win_min:.1f}, {shape_win_max:.1f}s]  "
              f"threshold={shape_threshold}SD  {cutoff_s}{skip_s}")