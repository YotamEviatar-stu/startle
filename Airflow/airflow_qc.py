import numpy as np
from scipy.signal import butter, filtfilt

MAD_TO_SD = 1.4826


def robust_sd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(np.abs(x - np.median(x))) * MAD_TO_SD)


def robust_cutoff(values, k):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return np.nan
    sd = robust_sd(v)
    if not (sd > 0):
        return np.nan
    return float(np.median(v) + k * sd)


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


def flag_excursions(signal_raw, sfreq, breath_size, k=5.0,
                    win_sec=1.0, merge_gap_sec=0.2, max_run_sec=1.0):
    x = np.asarray(signal_raw, dtype=np.float64).copy()
    n = len(x)
    mask = np.zeros(n, dtype=bool)
    if not (breath_size > 0):
        return x, mask, []

    bad = np.abs(x - _rolling_median(x, int(round(win_sec * sfreq)))) > k * breath_size

    raw_runs, i = [], 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        raw_runs.append([i, j])
        i = j

    gap = int(round(merge_gap_sec * sfreq))
    merged = []
    for r in raw_runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    cap = int(round(max_run_sec * sfreq))
    spans = []
    for i0, i1 in merged:
        if (i1 - i0) > cap:
            continue
        lo = x[i0 - 1] if i0 > 0 else (x[i1] if i1 < n else 0.0)
        hi = x[i1] if i1 < n else (x[i0 - 1] if i0 > 0 else 0.0)
        x[i0:i1] = np.linspace(lo, hi, (i1 - i0) + 2)[1:-1]
        mask[i0:i1] = True
        spans.append((i0 / sfreq, i1 / sfreq))
    return x, mask, spans


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
    A = session_breath_size(
        signal_raw, sfreq,
        band=getattr(config, "QC_BREATH_BAND", (0.01, 0.6)))
    clean, mask, spans = flag_excursions(
        signal_raw, sfreq, A,
        k=getattr(config, "QC_EXCURSION_K", 5.0),
        win_sec=getattr(config, "QC_EXCURSION_WIN_SEC", 1.0),
        merge_gap_sec=getattr(config, "QC_MERGE_GAP_SEC", 0.2),
        max_run_sec=getattr(config, "QC_MAX_RUN_SEC", 1.0))
    return {
        "breath_size": A,
        "signal_clean": clean,
        "mask": mask,
        "masked_spans": spans,
        "masked_sec": float(mask.sum() / sfreq),
    }


def cohort_breath_reference(sessions_cache, config):
    vals = {}
    for subj, sess_dict in sessions_cache.items():
        for sess_key, sess in sess_dict.items():
            vals[(subj, sess_key)] = session_breath_size(
                sess["signal_raw"], float(sess["sfreq"]),
                band=getattr(config, "QC_BREATH_BAND", (0.01, 0.6)))
    ref = float(np.median(list(vals.values()))) if vals else np.nan
    return ref, vals


def flat_session_report(sessions_cache, config):
    ref, vals = cohort_breath_reference(sessions_cache, config)
    ratio_min = getattr(config, "QC_FLAT_SESSION_RATIO", 0.10)
    rows = []
    for (subj, sess_key), A in sorted(vals.items(), key=lambda kv: kv[1]):
        frac = A / ref if ref > 0 else np.nan
        rows.append({
            "subject": subj, "session": sess_key, "breath_size": A,
            "frac_of_cohort": frac, "flat": bool(frac < ratio_min),
        })
    return ref, rows
