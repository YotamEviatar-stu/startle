import numpy as np
from scipy.signal import butter, filtfilt

MAD_TO_SD = 1.4826


def robust_sd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x))) * MAD_TO_SD)


def session_breath_size(signal_raw, sfreq, band=(0.01, 0.6)):
    x = np.asarray(signal_raw, dtype=np.float64)
    x = x - np.nanmean(x)
    nyq = sfreq / 2.0
    b, a = butter(1, [band[0] / nyq, band[1] / nyq], btype="band")
    return 2.0 * robust_sd(filtfilt(b, a, x))


def _rolling_median(x, win):
    win = win + 1 if win % 2 == 0 else win
    half = win // 2
    pad = np.pad(x, half, mode="edge")
    idx = np.arange(len(x))[:, None] + np.arange(win)[None, :]
    return np.median(pad[idx], axis=1)


GATE_ORDER = ("unmeasurable", "extreme", "rate_implausible", "no_inspiration", "lost_lock")


def classify_cycles(cycles, config):
    n = len(cycles)
    rejected = np.zeros(n, dtype=bool)
    reason = [None] * n

    ra = np.array([c["RA"] for c in cycles], dtype=float) if n else np.zeros(0)
    rp = np.array([c["RP"] for c in cycles], dtype=float) if n else np.zeros(0)
    usable = np.isfinite(ra) & (ra > 0)
    n_ref = int(usable.sum())

    min_ref = getattr(config, "CYCLE_MIN_REF_CYCLES", 30)
    scale   = getattr(config, "CYCLE_LOGRA_SCALE", 0.2378)
    k       = getattr(config, "CYCLE_LOGRA_K", 6.0)
    rp_max  = getattr(config, "CYCLE_RP_MAX", None)
    min_pk  = getattr(config, "CYCLE_LOSTLOCK_MIN_PEAKS", 3)

    abstained = []
    median_ra = float(np.median(ra[usable])) if n_ref else np.nan
    if n_ref >= min_ref and np.isfinite(median_ra) and median_ra > 0:
        cut = median_ra * float(np.exp(k * scale))
    else:
        cut = np.nan
        abstained.append("extreme")

    for i, c in enumerate(cycles):
        if not usable[i]:
            rejected[i], reason[i] = True, "unmeasurable"
        elif np.isfinite(cut) and ra[i] > cut:
            rejected[i], reason[i] = True, "extreme"
        elif rp_max is not None and rp[i] > rp_max:
            rejected[i], reason[i] = True, "rate_implausible"
        elif c.get("n_peaks") == 0:
            rejected[i], reason[i] = True, "no_inspiration"
        elif c.get("n_peaks", 0) >= min_pk:
            rejected[i], reason[i] = True, "lost_lock"

    report = _cycle_report(cycles, rejected, reason, abstained, n_ref, median_ra, cut)
    return rejected, reason, report


def _cycle_report(cycles, rejected, reason, abstained, n_ref, median_ra, cut):
    dur = np.array([c["RP"] for c in cycles], dtype=float) if len(cycles) else np.zeros(0)
    total_sec = float(dur.sum())
    counts, seconds = {}, {}
    for g in reason:
        if g is None:
            continue
        counts[g] = counts.get(g, 0) + 1
    for g in counts:
        seconds[g] = float(sum(d for d, r in zip(dur, reason) if r == g))
    kept = ~np.asarray(rejected, dtype=bool)
    return {
        "gate_counts": counts,
        "gate_seconds": seconds,
        "kept_frac": float(kept.mean()) if len(cycles) else np.nan,
        "kept_seconds_frac": float(dur[kept].sum() / total_sec) if total_sec > 0 else np.nan,
        "abstained": abstained,
        "n_ref_cycles": n_ref,
        "median_RA": median_ra,
        "cut_RA": cut,
        "n_cycles": len(cycles),
        "total_sec": total_sec,
    }


def apply_neighbour_rules(rejected, reason, config):
    rej = np.asarray(rejected, dtype=bool).copy()
    rsn = list(reason)
    n = len(rej)

    if getattr(config, "CYCLE_RECOVERY_RULE", True):
        for i in range(n - 1):
            if reason[i] == "extreme" and not rej[i + 1]:
                rej[i + 1], rsn[i + 1] = True, "recovery"

    if getattr(config, "CYCLE_GAPFILL_RULE", True):
        pre = rej.copy()
        for i in range(1, n - 1):
            if pre[i - 1] and pre[i + 1] and not pre[i]:
                rej[i], rsn[i] = True, "gap_fill"

    return rej, rsn


def flag_deep_breaths(cycles, max_ratio):
    n = len(cycles)
    if n < 3 or max_ratio is None:
        return np.zeros(n, dtype=bool)
    ra = np.array([c["RA"] for c in cycles], dtype=float)
    med = float(np.median(ra))
    if not (med > 0):
        return np.zeros(n, dtype=bool)
    return ra > (max_ratio * med)


def qc_session(signal_raw, sfreq, config):
    # The sample-level gate (flag_excursions) was deleted on 2026-08-15: there is
    # exactly one amplitude decision point and it is at the breath, in
    # classify_cycles, measured on signal_raw. Nothing is repaired or masked here,
    # so signal_clean IS signal_raw and there are no masked spans. This function
    # survives only to supply `breath_size` (A), still used for reporting.
    A = session_breath_size(
        signal_raw, sfreq,
        band=getattr(config, "QC_BREATH_BAND", (0.01, 0.6)))
    return {
        "breath_size": A,
        "signal_clean": np.asarray(signal_raw, dtype=float),
        "mask": np.zeros(len(signal_raw), dtype=bool),
        "masked_spans": [],
        "masked_sec": 0.0,
    }
