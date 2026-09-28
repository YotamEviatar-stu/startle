import numpy as np


def find_respiratory_extrema(resp, srate, custom_decision_threshold=0, sw_sizes=None):
    resp = np.asarray(resp, dtype=float).ravel()
    n = resp.size

    if sw_sizes is None:
        adj = srate / 1000.0
        sw_sizes = [int(np.floor(s * adj)) for s in (100, 300, 700, 1000, 5000)]
    sw_sizes = [int(s) for s in sw_sizes if int(s) >= 1]
    if not sw_sizes:
        raise ValueError("no usable sliding-window sizes for this sampling rate")

    pad_ind = int(min(n - 1, max(sw_sizes) * 2))
    padded = np.concatenate([resp, resp[n - pad_ind - 1:][::-1]])
    m = padded.size

    peak_threshold = resp.mean() + resp.std(ddof=1) / 2.0
    trough_threshold = resp.mean() - resp.std(ddof=1) / 2.0

    peak_vote = np.zeros(m, dtype=int)
    trough_vote = np.zeros(m, dtype=int)
    shifts = (1, 2, 3)

    for sw in sw_sizes:
        n_iters = m // sw - 1
        if n_iters < 1:
            continue
        for shift in shifts:
            start = sw - sw // shift
            seg = padded[start:start + n_iters * sw].reshape(n_iters, sw)
            rows = np.arange(n_iters)

            amax = seg.argmax(axis=1)
            keep = seg[rows, amax] > peak_threshold
            np.add.at(peak_vote, start + rows[keep] * sw + amax[keep], 1)

            amin = seg.argmin(axis=1)
            keep = seg[rows, amin] < trough_threshold
            np.add.at(trough_vote, start + rows[keep] * sw + amin[keep], 1)

    n_windows = len(sw_sizes) * len(shifts)
    thresholds = np.arange(1, n_windows + 1)
    n_peaks = np.array([(peak_vote > t).sum() for t in thresholds])
    n_troughs = np.array([(trough_vote > t).sum() for t in thresholds])

    if custom_decision_threshold > 0:
        decision = int(custom_decision_threshold)
    else:
        best_peak = int(np.argmax(np.diff(n_peaks))) + 1
        best_trough = int(np.argmax(np.diff(n_troughs))) + 1
        decision = int(np.floor((best_peak + best_trough) / 2.0))

    peaks = list(np.flatnonzero(peak_vote >= decision))
    troughs = list(np.flatnonzero(trough_vote >= decision))
    if not peaks or not troughs:
        return np.array([], dtype=int), np.array([], dtype=int)

    while troughs and peaks[0] > troughs[0]:
        troughs = troughs[1:]

    corrected_peaks = []
    corrected_troughs = []
    pki = 0
    tri = 0
    proceed = True

    while pki < len(peaks) - 2 and tri < len(troughs) - 2:
        peak_trough_diff = troughs[tri] - peaks[pki]
        peak_peak_diff = peaks[pki + 1] - peaks[pki]

        if peak_peak_diff < peak_trough_diff:
            if padded[peaks[pki + 1]] > padded[peaks[pki]]:
                pki += 1
            else:
                del peaks[pki + 1]
            proceed = False

        if proceed:
            trough_trough_diff = troughs[tri + 1] - troughs[tri]
            trough_peak_diff = peaks[pki + 1] - troughs[tri]
            if trough_trough_diff < trough_peak_diff:
                if padded[troughs[tri + 1]] < padded[troughs[tri]]:
                    tri += 1
                else:
                    del troughs[tri + 1]
                proceed = False

        if proceed:
            if peak_trough_diff > 0:
                corrected_peaks.append(peaks[pki])
                corrected_troughs.append(troughs[tri])
                tri += 1
                pki += 1
            else:
                raise RuntimeError(
                    "peaks got ahead of troughs at peak %d / trough %d"
                    % (peaks[pki], troughs[tri])
                )
        proceed = True

    cp = np.array([p for p in corrected_peaks if p < n - 1], dtype=int)
    ct = np.array([t for t in corrected_troughs if t < n - 1], dtype=int)
    return cp, ct
