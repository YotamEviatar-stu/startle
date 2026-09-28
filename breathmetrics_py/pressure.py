import numpy as np
from scipy.signal import butter, sosfiltfilt

from .fft_smooth import fft_smooth


def lowpass(x, srate, cutoff_hz, order=4):
    x = np.asarray(x, dtype=float).ravel()
    if cutoff_hz is None or cutoff_hz <= 0:
        return x.copy()
    nyq = srate / 2.0
    if cutoff_hz >= nyq:
        return x.copy()
    sos = butter(order, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def estimate_zero(x, srate, method="histogram", win_sec=60.0, nbins=200,
                  percentile=50.0):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size

    if method == "simple":
        return np.full(n, x.mean())
    if method == "sliding":
        return fft_smooth(x, int(round(srate * win_sec)))
    if method not in ("histogram", "percentile"):
        raise ValueError("unknown zero_method: %r" % method)

    win = int(round(win_sec * srate))
    win = max(min(win, n), 1)
    step = max(win // 2, 1)
    starts = np.arange(0, max(n - win, 0) + 1, step)
    if starts.size == 0:
        starts = np.array([0])

    centers = []
    values = []
    for s in starts:
        seg = x[s:s + win]
        if method == "percentile":
            v = np.percentile(seg, percentile)
        else:
            lo, hi = np.percentile(seg, [1.0, 99.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                v = float(np.median(seg))
            else:
                counts, edges = np.histogram(seg, bins=nbins, range=(lo, hi))
                k = int(np.argmax(counts))
                v = 0.5 * (edges[k] + edges[k + 1])
        centers.append(s + seg.size / 2.0)
        values.append(v)

    if len(values) == 1:
        return np.full(n, values[0])
    return np.interp(np.arange(n, dtype=float), np.asarray(centers, dtype=float),
                     np.asarray(values, dtype=float))


def pressure_to_flow(pressure, srate, transform="sqrt", zero_method="histogram",
                     zero_win_sec=60.0, zero_nbins=200, zero_percentile=50.0,
                     lowpass_hz=4.0, lowpass_order=4):
    p = np.asarray(pressure, dtype=float).ravel()

    filtered = lowpass(p, srate, lowpass_hz, lowpass_order)
    zero = estimate_zero(filtered, srate, zero_method, zero_win_sec,
                         zero_nbins, zero_percentile)

    d = filtered - zero

    if transform == "none":
        flow = d
    elif transform == "sqrt":
        flow = np.sign(d) * np.sqrt(np.abs(d))
    else:
        raise ValueError("unknown transform: %r" % transform)

    info = {
        "zero": zero,
        "filtered_pressure": filtered,
        "transform": transform,
    }
    return flow, info
