import numpy as np


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else np.nan


def _nanstd(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    if a.size == 1:
        return 0.0
    return float(np.std(a, ddof=1))


def _exclude_outliers(values, valid_inds):
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values)
    sd = _nanstd(values)
    keep = (values > mu - 2 * sd) & (values < mu + 2 * sd)
    mask = np.zeros(values.size, dtype=bool)
    mask[np.asarray(valid_inds, dtype=int)] = True
    return values[keep & mask]


def get_secondary_features(bm):
    n_inhales = bm.inhale_onsets.size
    n_exhales = bm.exhale_onsets.size

    if bm.statuses is None:
        valid_inhale_inds = np.arange(n_inhales)
        valid_exhale_inds = np.arange(n_exhales)
    else:
        rejected = np.flatnonzero(np.asarray(bm.statuses) == "rejected")
        valid_inhale_inds = np.setdiff1d(np.arange(n_inhales), rejected)
        valid_exhale_inds = np.setdiff1d(np.arange(n_exhales), rejected)

    n_valid_inhales = valid_inhale_inds.size
    n_valid_exhales = valid_exhale_inds.size

    breath_diffs = []
    for i in range(n_valid_inhales - 1):
        this_b = valid_inhale_inds[i]
        next_b = valid_inhale_inds[i + 1]
        if next_b == this_b + 1:
            breath_diffs.append(bm.inhale_onsets[next_b] - bm.inhale_onsets[this_b])
    breath_diffs = np.asarray(breath_diffs, dtype=float)

    breathing_rate = bm.srate / np.mean(breath_diffs) if breath_diffs.size else np.nan
    ibi = 1.0 / breathing_rate if breathing_rate else np.nan
    cv_breathing_rate = (_nanstd(breath_diffs) / np.mean(breath_diffs)
                         if breath_diffs.size else np.nan)

    out = {
        "Breathing Rate": breathing_rate,
        "Average Inter-Breath Interval": ibi,
        "Coefficient of Variation of Breathing Rate": cv_breathing_rate,
    }

    if bm.data_type not in ("humanAirflow", "rodentAirflow"):
        return out

    valid_inhale_flows = _exclude_outliers(bm.peak_inspiratory_flows, valid_inhale_inds)
    valid_exhale_flows = _exclude_outliers(bm.trough_expiratory_flows, valid_exhale_inds)
    valid_inhale_volumes = _exclude_outliers(bm.inhale_volumes, valid_inhale_inds)
    valid_exhale_volumes = _exclude_outliers(bm.exhale_volumes, valid_exhale_inds)

    avg_inhale_volume = _nanmean(valid_inhale_volumes)
    avg_exhale_volume = _nanmean(valid_exhale_volumes)

    avg_inhale_duration = _nanmean(bm.inhale_durations)
    avg_exhale_duration = _nanmean(bm.exhale_durations)

    pct_inhale_pause = np.sum(~np.isnan(bm.inhale_pause_durations)) / max(n_valid_inhales, 1)
    pct_exhale_pause = np.sum(~np.isnan(bm.exhale_pause_durations)) / max(n_valid_exhales, 1)

    avg_inhale_pause = _nanmean(bm.inhale_pause_durations[valid_inhale_inds]) * pct_inhale_pause
    avg_exhale_pause = _nanmean(bm.exhale_pause_durations[valid_exhale_inds]) * pct_exhale_pause
    if np.isnan(avg_inhale_pause):
        avg_inhale_pause = 0.0
    if np.isnan(avg_exhale_pause):
        avg_exhale_pause = 0.0

    out.update({
        "Average Peak Inspiratory Flow": _nanmean(valid_inhale_flows),
        "Average Peak Expiratory Flow": _nanmean(valid_exhale_flows),
        "Average Inhale Volume": avg_inhale_volume,
        "Average Exhale Volume": avg_exhale_volume,
        "Average Tidal Volume": avg_inhale_volume + avg_exhale_volume,
        "Minute Ventilation": breathing_rate * (avg_inhale_volume + avg_exhale_volume),
        "Duty Cycle of Inhale": avg_inhale_duration / ibi,
        "Duty Cycle of Inhale Pause": avg_inhale_pause / ibi,
        "Duty Cycle of Exhale": avg_exhale_duration / ibi,
        "Duty Cycle of Exhale Pause": avg_exhale_pause / ibi,
        "Coefficient of Variation of Inhale Duty Cycle": _nanstd(bm.inhale_durations) / avg_inhale_duration,
        "Coefficient of Variation of Exhale Duty Cycle": _nanstd(bm.exhale_durations) / avg_exhale_duration,
        "Average Inhale Duration": avg_inhale_duration,
        "Average Inhale Pause Duration": avg_inhale_pause,
        "Average Exhale Duration": avg_exhale_duration,
        "Average Exhale Pause Duration": avg_exhale_pause,
        "Percent of Breaths With Inhale Pause": pct_inhale_pause,
        "Percent of Breaths With Exhale Pause": pct_exhale_pause,
        "Coefficient of Variation of Breath Volumes": (
            _nanstd(valid_inhale_volumes) / avg_inhale_volume
            if valid_inhale_volumes.size else np.nan),
    })
    return out
