import numpy as np

MAXIMUM_BIN_THRESHOLD = 5.0
BINNING_THRESHOLD = 0.25


def _hist_centers(x, centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        return np.array([x.size]), centers
    edges = np.concatenate(
        ([-np.inf], (centers[:-1] + centers[1:]) / 2.0, [np.inf])
    )
    counts, _ = np.histogram(x, bins=edges)
    return counts, centers


def _round_half_up(v):
    return int(np.floor(v + 0.5))


def _pause_range(counts, centers, mode_bin, max_pause_bins):
    lo = centers[mode_bin]
    hi = centers[min(mode_bin + 1, centers.size - 1)]
    mode_total = counts[mode_bin]
    for k in range(1, max_pause_bins + 1):
        j = mode_bin - k
        if j < 0:
            break
        if counts[j] > mode_total * BINNING_THRESHOLD:
            lo = centers[j]
    for k in range(1, max_pause_bins + 1):
        j = mode_bin + k
        if j >= centers.size:
            break
        if counts[j] > mode_total * BINNING_THRESHOLD:
            hi = centers[j]
    return lo, hi


def find_respiratory_pauses_and_onsets(resp, peaks, troughs, n_bins=100):
    resp = np.asarray(resp, dtype=float).ravel()
    peaks = np.asarray(peaks, dtype=int).ravel()
    troughs = np.asarray(troughs, dtype=int).ravel()

    inhale_onsets = np.zeros(peaks.size, dtype=int)
    exhale_onsets = np.zeros(troughs.size, dtype=int)
    inhale_pause_onsets = np.full(peaks.size, np.nan)
    exhale_pause_onsets = np.full(troughs.size, np.nan)

    max_pause_bins = 5 if n_bins >= 100 else 2
    upper_threshold = _round_half_up(n_bins * 0.7)
    lower_threshold = _round_half_up(n_bins * 0.3)
    simple_zero_cross = resp.mean()

    tail_onset_lims = int(np.floor(np.mean(np.diff(peaks)))) if peaks.size > 1 else peaks[0]

    first_boundary = peaks[0] - tail_onset_lims if peaks[0] > tail_onset_lims else 0

    window = resp[first_boundary:peaks[0] + 1]
    centers = np.linspace(window.min(), window.max(), n_bins)
    counts, centers = _hist_centers(window, centers)
    mode_bin = int(np.argmax(counts))
    zero_cross_threshold = centers[mode_bin]
    if (mode_bin + 1) < lower_threshold or (mode_bin + 1) > upper_threshold:
        zero_cross_threshold = simple_zero_cross

    below = np.flatnonzero(window < zero_cross_threshold)
    inhale_onsets[0] = first_boundary + below[-1] + 1 if below.size else first_boundary

    for b in range(peaks.size - 1):
        tr = troughs[b]
        inhale_window = resp[tr:peaks[b + 1] + 1]
        centers = np.linspace(inhale_window.min(), inhale_window.max(), n_bins)
        counts, centers = _hist_centers(inhale_window, centers)
        mode_bin = int(np.argmax(counts))
        max_bin_ratio = counts[mode_bin] / counts.mean()
        is_exhale_pause = not (
            (mode_bin + 1) < lower_threshold
            or (mode_bin + 1) > upper_threshold
            or max_bin_ratio < MAXIMUM_BIN_THRESHOLD
        )

        if not is_exhale_pause:
            not_above = np.flatnonzero(~(inhale_window > simple_zero_cross))
            exhale_pause_onsets[b] = np.nan
            inhale_onsets[b + 1] = tr + (not_above[-1] + 1 if not_above.size else 0)
        else:
            lo, hi = _pause_range(counts, centers, mode_bin, max_pause_bins)
            putative = np.flatnonzero((inhale_window > lo) & (inhale_window < hi))
            exhale_pause_onsets[b] = tr + putative[0]
            inhale_onsets[b + 1] = tr + putative[-1] + 2

        pk = peaks[b]
        exhale_window = resp[pk:tr + 1]
        centers = np.linspace(exhale_window.min(), exhale_window.max(), n_bins)
        counts, centers = _hist_centers(exhale_window, centers)
        mode_bin = int(np.argmax(counts))
        max_bin_ratio = counts[mode_bin] / counts.mean()
        is_inhale_pause = not (
            (mode_bin + 1) < lower_threshold
            or (mode_bin + 1) > upper_threshold
            or max_bin_ratio < MAXIMUM_BIN_THRESHOLD
        )

        if not is_inhale_pause:
            above = np.flatnonzero(exhale_window > simple_zero_cross)
            inhale_pause_onsets[b] = np.nan
            exhale_onsets[b] = pk + (above[-1] + 1 if above.size else 0)
        else:
            lo, hi = _pause_range(counts, centers, mode_bin, 5)
            putative = np.flatnonzero((exhale_window > lo) & (exhale_window < hi))
            inhale_pause_onsets[b] = pk + putative[0]
            exhale_onsets[b] = pk + putative[-1] + 2

    last_peak = peaks[-1]
    if resp.size - last_peak > tail_onset_lims:
        last_boundary = last_peak + tail_onset_lims
    else:
        last_boundary = resp.size - 1

    exhale_window = resp[last_peak:last_boundary + 1]
    below = np.flatnonzero(exhale_window < simple_zero_cross)
    if below.size:
        exhale_onsets[-1] = last_peak + below[0] + 1
    else:
        exhale_onsets[-1] = last_boundary

    return inhale_onsets, exhale_onsets, inhale_pause_onsets, exhale_pause_onsets
