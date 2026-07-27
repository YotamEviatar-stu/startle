# hr_demo_processor.py — stateless processing for the HR pipeline (Demo/Testing).
#
# ── Cache architecture ─────────────────────────────────────────────────────────
#
# Layer 1  process_session()       Handled by hr_processor.py — run hr_main.py to
#                                  populate the cache. No Layer 1 code lives here.
#
# Layer 2  apply_analysis_params() Everything else:
#            • Epoch cutting at current WIDE_TMIN / WIDE_TMAX
#            • Artifact cleaning (flat plateaus + spikes)
#            • Baseline, rejection, scoring
#            • Shape-centroid outlier rejection (extra gate vs hr_processor)
#          Returns (trials_data, traces). Never touches the MFF.
#
# What requires FORCE_RELOAD (Layer 1 changes):
#   HR_CHANNEL, HR_CACHE_SFREQ, trigger codes
#
# What is free to change in the notebook (Layer 2):
#   WIDE_TMIN/TMAX, BASELINE_*, SCORE_*, all rejection gates,
#   HR_SHAPE_*, USE_SUBJECTIVE_TRIAL_TYPE
#
# ── Signal notes ──────────────────────────────────────────────────────────────
#
# SpO2-Pulse is a hardware-computed BPM staircase (~2 s steps). It is NOT a
# continuous waveform — no filter should be applied before scoring unless the
# user explicitly enables SIGNAL_LOWPASS_HZ. Artifact cleaning (_clean_epoch)
# interpolates dropout plateaus and spike artifacts before any analysis.
#
# Shape-centroid rejection compares each trial's z-scored BPM epoch to the
# session mean shape. This catches structurally atypical trials (persistent
# plateau, monotonic drift, disconnects) that pass the baseline-only gates
# because the baseline window itself looked normal.

import numpy as np
from scipy.signal import butter, filtfilt

from HR.hr_processor import _clean_epoch, _window_artifact_fractions


# ── Layer 2: analysis (fast, no MFF access) ───────────────────────────────────

def apply_analysis_params(sessions_cache, config):
    """
    Run epoch cutting + artifact cleaning + baseline + rejection + scoring.
    Adds shape-centroid outlier rejection on top of the standard hr_processor gates.

    Accepts two cache formats (same as hr_processor.apply_analysis_params):
      NEW  {subj: {sess_key: session_dict with raw_signal, startle_samps, ...}}
           → re-cuts epochs from raw signal (WIDE_TMIN/TMAX tunable)
      OLD  {subj: {sess_key: [list of trial dicts with epoch_wide pre-cut]}}
           → runs analysis on existing epochs (WIDE_TMIN/TMAX fixed at cache time)

    Returns (trials_data, traces) — same nested format as hr_processor so
    hr_raw_show.ipynb can swap this processor in by changing one import line.
    """
    trials_data = {}
    traces      = {}

    for subj, sess_dict in sorted(sessions_cache.items()):
        trials_data[subj] = {}
        traces[subj]      = {}

        for sess_key, sess in sess_dict.items():

            # ── Old format: sess is already a list of trial dicts ──────────────
            if isinstance(sess, list):
                trials = [dict(t) for t in sess]
                for t in trials:
                    t["baseline_mean"]    = None
                    t["score"]            = None
                    t["rejected"]         = None
                    t["rejection_reason"] = ""
                    t["epoch_anal"]       = None
                    t["times_anal"]       = None
                    t["artifact_mask"]    = None
                    t["any_artifact"]     = False
                    t.pop("shape_msd",    None)
                    t.pop("shape_cutoff", None)
                _run_hr_demo_analysis(trials, config)
                trials_data[subj][sess_key] = trials
                traces[subj][sess_key]      = {}
                continue

            # ── New format: sess has raw_signal + startle_samps + trials_meta ──
            raw_sig       = sess["raw_signal"]
            sfreq         = sess["sfreq"]
            startle_samps = sess["startle_samps"]
            trials_meta   = sess["trials_meta"]

            traces[subj][sess_key] = {
                "signal":            raw_sig,
                "sfreq":             sfreq,
                "startle_samps_sec": startle_samps / sfreq,
            }

            wide_n     = int((config.WIDE_TMAX - config.WIDE_TMIN) * sfreq)
            times_wide = np.linspace(config.WIDE_TMIN, config.WIDE_TMAX,
                                     wide_n, endpoint=False)

            trials = []
            for star_samp, meta in zip(startle_samps, trials_meta):
                t0 = int(star_samp + config.WIDE_TMIN * sfreq)
                t1 = t0 + wide_n
                if t0 < 0 or t1 > len(raw_sig):
                    continue
                trials.append({
                    "epoch_wide":       raw_sig[t0:t1].copy(),
                    "times_wide":       times_wide,
                    "sfreq":            sfreq,
                    "channel":          sess["channel"],
                    "subjective_label": meta["subjective_label"],
                    "label":            meta["subjective_label"],
                    "valence":          meta["valence"],
                    "arousal":          meta["arousal"],
                    "image_detail":     meta["image_detail"],
                    "image_type":       meta.get("image_type", ""),
                    "baseline_mean":    None,
                    "score":            None,
                    "rejected":         None,
                    "rejection_reason": "",
                    "epoch_anal":       None,
                    "times_anal":       None,
                })

            _run_hr_demo_analysis(trials, config)
            trials_data[subj][sess_key] = trials

    return trials_data, traces


# ── Internal: full analysis with shape-centroid rejection ─────────────────────

def _run_hr_demo_analysis(trials, config):
    """HR analysis + shape-centroid outlier rejection.

    Processing order:
      Pass 1 — label resolution, artifact cleaning, baseline, session-level statistics
      Shape  — centroid distance matrix (see detailed block below)
      Pass 2 — all rejection gates, scoring, epoch trim, summary log
    """
    if not trials:
        return

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 1  Labels · artifact cleaning · baseline · session stats
    # ══════════════════════════════════════════════════════════════════════════
    bl_stds   = []
    pre_masks = []

    for t in trials:
        # ── Condition label ──────────────────────────────────────────────────
        if config.USE_SUBJECTIVE_TRIAL_TYPE:
            t["label"] = t["subjective_label"]
        else:
            img = (str(t.get("image_type") or t.get("image_detail", ""))).lower().strip()
            if "negative" in img or img == "1":
                t["label"] = 1
            elif "neutral" in img or img == "2":
                t["label"] = 2
            else:
                t["label"] = t["subjective_label"]

        # ── Artifact cleaning ────────────────────────────────────────────────
        ep_clean, art_mask = _clean_epoch(
            t["epoch_wide"],
            t["sfreq"],
            flat_min_sec     = getattr(config, "FLAT_MIN_SEC",     3.0),
            spike_thresh_bpm = getattr(config, "SPIKE_THRESH_BPM", 20.0),
        )
        t["epoch_wide"]   = ep_clean
        t["artifact_mask"] = art_mask
        t["any_artifact"]  = bool(np.any(art_mask))

        # ── Optional lowpass smoothing ────────────────────────────────────────
        lp_hz = getattr(config, "SIGNAL_LOWPASS_HZ", None)
        if lp_hz is not None and lp_hz > 0:
            sfreq = t["sfreq"]
            nyq   = sfreq / 2.0
            if lp_hz < nyq:
                b, a = butter(4, lp_hz / nyq, btype="low")
                t["epoch_filtered"] = filtfilt(b, a, ep_clean)
            else:
                t["epoch_filtered"] = ep_clean.copy()
        else:
            t["epoch_filtered"] = ep_clean.copy()

        # ── Baseline stats ───────────────────────────────────────────────────
        tw       = t["times_wide"]
        ep       = t["epoch_filtered"]
        pre_mask = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
        pre_masks.append(pre_mask)

        if np.any(pre_mask):
            bm = float(np.mean(ep[pre_mask]) if config.BASELINE_METHOD == "mean"
                       else np.median(ep[pre_mask]))
            bs = float(np.std(ep[pre_mask]))
        else:
            bm = bs = np.nan
        t["baseline_mean"] = bm
        t["baseline_std"]  = bs
        bl_stds.append(bs)

    sess_std_median = float(np.nanmedian(bl_stds))
    sess_std_std    = float(np.nanstd(bl_stds))

    # ══════════════════════════════════════════════════════════════════════════
    # SHAPE-CENTROID OUTLIER DETECTION
    #
    # Principle: a trial whose BPM trajectory deviates strongly from the
    # session's average HR pattern is flagged as structurally atypical.
    # This catches post-stimulus plateaus, monotonic drift, and gross
    # probe-off artifacts that share nothing in common with the rest of the
    # session — artifacts that slip past the baseline-only gates because the
    # baseline window itself may look normal.
    #
    # Config parameters (all read from the config object):
    #   HR_SHAPE_REJECTION_ENABLE  bool   master switch (default True)
    #   HR_SHAPE_SD_THRESHOLD      float  cutoff = mean(MSD) + k×std(MSD)
    #                                     (default 3.0 SDs above mean)
    #   HR_SHAPE_WINDOW_MIN        float  window start, seconds re t=0
    #   HR_SHAPE_WINDOW_MAX        float  window end,   seconds re t=0
    #
    # The z-score normalisation step (row_means, row_stds) makes the comparison
    # amplitude-independent: a subject averaging 55 BPM and one averaging 80 BPM
    # produce comparable shape vectors. Flatlines (std = 0) → row stays as zeros
    # → maximum distance from centroid, so genuine flatlines are still caught.
    # ══════════════════════════════════════════════════════════════════════════

    shape_enable    = getattr(config, "HR_SHAPE_REJECTION_ENABLE", True)
    shape_threshold = getattr(config, "HR_SHAPE_SD_THRESHOLD",     3.0)
    shape_win_min   = getattr(config, "HR_SHAPE_WINDOW_MIN",
                              getattr(config, "ANAL_TMIN",  -5.0))
    shape_win_max   = getattr(config, "HR_SHAPE_WINDOW_MAX",
                              getattr(config, "SCORE_TMAX",  5.0))

    # Per-trial MSD score and session-level cutoff — initialised to nan so
    # Pass 2 can test for them cleanly even when the block is skipped.
    shape_msd    = [np.nan] * len(trials)
    shape_cutoff = np.nan

    if shape_enable and len(trials) > 3:

        # ── Phase 1: Matrix Alignment ────────────────────────────────────────
        # Extract the shape window from every trial's cleaned BPM epoch.
        # Result: a 2-D matrix of shape (n_trials × n_samples_in_window).
        tw_ref     = trials[0]["times_wide"]
        shape_mask = (tw_ref >= shape_win_min) & (tw_ref < shape_win_max)
        n_cols     = int(np.sum(shape_mask))

        if n_cols > 0:
            waveform_matrix = np.vstack(
                [t["epoch_filtered"][shape_mask].astype(np.float64)
                 for t in trials]
            )                                               # (n_trials × n_cols)

            # ── Z-score normalise each row (amplitude-independent shape comparison) ──
            # Subtract each trial's mean and divide by its std so that a subject
            # resting at 55 BPM and one at 80 BPM produce comparable shape vectors.
            # Flatlines (std = 0) → row stays as zeros → maximum distance from
            # centroid, so genuine flatlines are still caught.
            row_means = waveform_matrix.mean(axis=1, keepdims=True)
            row_stds  = waveform_matrix.std(axis=1, keepdims=True)
            row_stds[row_stds == 0] = 1.0                  # avoid div-by-zero
            waveform_matrix = (waveform_matrix - row_means) / row_stds

            # ── Phase 2: Centroid Vector Generation ──────────────────────────
            centroid = np.mean(waveform_matrix, axis=0)    # (n_cols,)

            # ── Phase 3: Outlier Evaluation via Mean Squared Distance ─────────
            msd_per_trial = np.mean(
                (waveform_matrix - centroid) ** 2, axis=1  # (n_trials,)
            )

            msd_mean     = float(np.nanmean(msd_per_trial))
            msd_std      = float(np.nanstd(msd_per_trial))
            shape_cutoff = msd_mean + shape_threshold * msd_std

            shape_msd = msd_per_trial.tolist()

            # Attach per-trial diagnostics for notebook inspection
            for t, msd in zip(trials, shape_msd):
                t["shape_msd"]    = float(msd)
                t["shape_cutoff"] = float(shape_cutoff)

    # ══════════════════════════════════════════════════════════════════════════
    # PASS 2  Rejection gates · scoring · epoch trim
    # ══════════════════════════════════════════════════════════════════════════
    for t, pre_mask, msd in zip(trials, pre_masks, shape_msd):
        tw  = t["times_wide"]
        ep  = t["epoch_filtered"]
        bm  = t["baseline_mean"]
        bs  = t["baseline_std"]

        # Gate: BPM range (probe off or implausible signal)
        range_reject = np.isnan(bm) or bm < config.HR_MIN_BPM or bm > config.HR_MAX_BPM

        # Gate: noisy baseline z-score
        z_reject  = False
        threshold = getattr(config, "HR_Z_SCORE_THRESHOLD", None)
        if threshold is not None and not np.isnan(bs) and sess_std_std > 0:
            z_reject = bool((bs - sess_std_median) / sess_std_std > threshold)

        # Gate: artifact coverage in baseline window
        art_mask = t.get("artifact_mask", np.zeros(len(tw), dtype=bool))
        bl_mask  = (tw >= config.BASELINE_TMIN) & (tw < config.BASELINE_TMAX)
        bl_total, bl_contig = _window_artifact_fractions(art_mask, bl_mask)
        bl_reject = (
            bl_total  > getattr(config, "FLAT_BASELINE_TOTAL_MAX",      0.50) or
            bl_contig > getattr(config, "FLAT_BASELINE_CONTIGUOUS_MAX", 0.40)
        )

        # Gate: artifact coverage in score window
        sc_mask  = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
        sc_total, sc_contig = _window_artifact_fractions(art_mask, sc_mask)
        sc_reject = (
            sc_total  > getattr(config, "FLAT_SCORE_TOTAL_MAX",      0.50) or
            sc_contig > getattr(config, "FLAT_SCORE_CONTIGUOUS_MAX", 0.40)
        )

        # Gate: dominant flat block in score window (BPM never updated)
        sc_dominant_reject = False
        sc_dom_thresh = getattr(config, "SCORE_FLAT_DOMINANT_MAX", None)
        if sc_dom_thresh is not None and np.any(sc_mask):
            sc_ep    = t["epoch_wide"][sc_mask]   # raw staircase, not interpolated
            changes  = np.concatenate([[0], np.where(np.diff(sc_ep) != 0)[0] + 1, [len(sc_ep)]])
            dominant_frac = float(np.max(np.diff(changes))) / len(sc_ep)
            sc_dominant_reject = dominant_frac > sc_dom_thresh

        # Gate: shape-centroid outlier
        shape_reject = (
            shape_enable
            and not np.isnan(msd)
            and not np.isnan(shape_cutoff)
            and bool(msd > shape_cutoff)
        )

        reasons = []
        if range_reject:       reasons.append("bpm_range")
        if z_reject:           reasons.append("noisy_baseline")
        if bl_reject:          reasons.append("flat_baseline")
        if sc_reject:          reasons.append("flat_score_window")
        if sc_dominant_reject: reasons.append("flat_score_dominant")
        if shape_reject:       reasons.append("atypical_shape")

        t["rejected"]         = bool(reasons)
        t["rejection_reason"] = ", ".join(reasons) if reasons else ""

        # Epoch trim
        anal_mask       = (tw >= config.ANAL_TMIN) & (tw < config.ANAL_TMAX)
        t["epoch_anal"] = ep[anal_mask]
        t["times_anal"] = tw[anal_mask]

        # Scoring
        if t["rejected"]:
            t["score"] = np.nan
        elif config.SCORE_METHOD == "max_minus_baseline":
            score_mask = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
            if np.any(score_mask) and not np.isnan(bm) and bm != 0:
                t["score"] = float((np.max(ep[score_mask]) - bm) / bm * 100)
            else:
                t["score"] = np.nan
        elif config.SCORE_METHOD == "mean_minus_baseline":
            score_mask = (tw >= config.SCORE_TMIN) & (tw <= config.SCORE_TMAX)
            if np.any(score_mask) and not np.isnan(bm) and bm != 0:
                t["score"] = float((np.mean(ep[score_mask]) - bm) / bm * 100)
            else:
                t["score"] = np.nan
        else:
            t["score"] = np.nan

        # Gate: score ceiling (applied post-scoring)
        score_max = getattr(config, "SCORE_MAX_PCT", None)
        if score_max is not None and not t["rejected"] and not np.isnan(t["score"]):
            if abs(t["score"]) > score_max:
                t["rejected"]         = True
                t["score"]            = np.nan
                t["rejection_reason"] = "score_too_large"

        # Gate: score floor (applied post-scoring)
        score_min = getattr(config, "SCORE_MIN_PCT", None)
        if score_min is not None and not t["rejected"] and not np.isnan(t["score"]):
            if t["score"] < score_min:
                t["rejected"]         = True
                t["score"]            = np.nan
                t["rejection_reason"] = "score_negative"

    # ── Shape-centroid summary ────────────────────────────────────────────────
    if shape_enable:
        n_shape  = sum(1 for t in trials
                       if "atypical_shape" in (t.get("rejection_reason") or ""))
        n_total  = len(trials)
        cutoff_s = (f"MSD cutoff={shape_cutoff:.3e}"
                    if not np.isnan(shape_cutoff) else "gate skipped (<4 trials)")
        print(f"  [shape-centroid] {n_shape}/{n_total} trials flagged as atypical_shape  "
              f"window=[{shape_win_min:.1f}, {shape_win_max:.1f}s]  "
              f"threshold={shape_threshold}SD  {cutoff_s}")
