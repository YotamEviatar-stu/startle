import numpy as np
from scipy.signal import oaconvolve


def mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def prctile(x, p):
    x = np.sort(x[~np.isnan(x)])
    n = x.size
    if n == 0:
        return np.nan
    q = 100 * (np.arange(1, n + 1) - 0.5) / n
    if p <= q[0]:
        return x[0]
    if p >= q[-1]:
        return x[-1]
    return np.interp(p, q, x)


def firws_kernel(order, fc, fs, high):
    m = np.arange(-order // 2, order // 2 + 1)
    f = fc / fs
    b = np.where(m == 0, 2 * np.pi * f, np.sin(2 * np.pi * f * m) / np.where(m == 0, 1, m))
    b = b * (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(order + 1) / order))
    b = b / b.sum()
    if high:
        b = -b
        b[order // 2] += 1
    return b


def firws_order(df, fs):
    return int(np.ceil((3.3 / (df / fs)) / 2) * 2)


def fir_df(fc, fs):
    max_df = min(fc * 2, (fs / 2 - fc) * 2)
    return min(max(fc * 0.25, 2), max_df)


def ft_firws(x, fs, fc, high):
    b = firws_kernel(firws_order(fir_df(fc, fs), fs), fc, fs, high)
    gd = (b.size - 1) // 2
    mu = np.nanmean(x)
    x = x - mu
    y = oaconvolve(np.concatenate([np.full(gd, x[0]), x, np.full(gd, x[-1])]), b, mode="valid")
    return y if high else y + mu


def peakseek(x, minpeakh, minpeakdist):
    locs = np.flatnonzero((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:])) + 1
    locs = locs[~(x[locs] <= minpeakh)]
    if minpeakdist > 1:
        while True:
            close = np.diff(locs) < minpeakdist
            if not close.any():
                break
            pks = x[locs]
            left, right = pks[:-1][close], pks[1:][close]
            deln = np.flatnonzero(close)
            first = left <= right
            locs = np.delete(locs, np.concatenate([deln[first], deln[~first] + 1]))
    return locs


def hpd_skewness(x):
    return np.nanmean(x ** 3) / np.nanmean(x ** 2) ** 1.5


def ft_kurtosis(x):
    x0 = x - np.mean(x)
    return np.mean(x0 ** 4) / np.mean(x0 ** 2) ** 2


def heart_peak_detect(sig, fs, corthresh=0.2, thresh=10.0, mindist=0.35,
                      hpfreq=1.0, lpfreq=100.0, PRmax=0.25, QRmax=0.05, RSmax=0.1, QTmax=0.42):
    ecg = ft_firws(ft_firws(np.asarray(sig, float), fs, lpfreq, False), fs, hpfreq, True)
    n = ecg.size
    z2 = ((ecg - ecg.mean()) / ecg.std(ddof=1)) ** 2
    r = peakseek(z2, thresh, fs * mindist)

    hb = int(mround(0.5 * fs))
    rows = [ecg[r0 - hb:r0 + hb + 1] for r0 in r if (r0 + 1) - hb > 1 and (r0 + 1) + hb < n]
    mhb = np.mean(rows, axis=0) if rows else np.full(2 * hb + 1, np.nan)

    if np.sign(hpd_skewness(ecg)) == -1:
        ecg, mhb = -ecg, -mhb

    pad = np.concatenate([np.zeros(1000), ecg, np.zeros(1000)])
    pad[np.isnan(pad)] = 0
    L = mhb.size
    cr = np.zeros(pad.size)
    off = int(mround(L / 2)) - 1
    cr[off:off + pad.size - L] = (np.correlate(pad, mhb, mode="valid") / L)[:pad.size - L]
    cr = cr / np.max(cr)
    crc = cr[1000:-1000]
    r = peakseek(crc, corthresh, fs * mindist)

    beats = dict(P=[], Q=[], R=[], S=[], T=[])
    for r0 in r:
        r1 = r0 + 1
        s = int(max(1, mround(r1 - QRmax * fs)))
        q1 = s + int(np.argmin(ecg[s - 1:r1]))
        s = int(max(1, mround(r1 - PRmax * fs)))
        p1 = s + int(np.argmax(ecg[s - 1:q1]))
        e = int(min(n, mround(r1 + RSmax * fs)))
        s1 = r1 + int(np.argmin(ecg[r1 - 1:e]))
        e = int(min(n, mround(q1 + QTmax * fs)))
        t1 = s1 + int(np.argmax(ecg[s1 - 1:e]))
        for k, v in zip("PQRST", (p1, q1, r1, s1, t1)):
            beats[k].append(v - 1)
    beats = {k: np.array(v, dtype=int) for k, v in beats.items()}
    return beats, hpd_skewness(crc), ft_kurtosis(crc)


def cv(x):
    if x.size == 0:
        return np.nan
    return (np.std(x, ddof=1) if x.size > 1 else 0.0) / abs(np.mean(x))


def caracas(sources, fs, bpm_min=35, bpm_max=90):
    dur_min = sources.shape[1] / fs / 60
    out = []
    for tc in sources:
        beats, sk, ku = heart_peak_detect(tc, fs)
        rr = np.diff(beats["R"]) / fs
        rr_f = rr[(rr >= prctile(rr, 0)) & (rr <= prctile(rr, 70))] if rr.size else rr
        ra = np.abs(tc[beats["R"]])
        ra_f = ra[(ra >= prctile(ra, 15)) & (ra <= prctile(ra, 85))] if ra.size else ra
        m = dict(sk=sk, ku=ku, RR=cv(rr_f), Rampl=cv(ra_f), nbeats=beats["R"].size,
                 bpm=beats["R"].size / dur_min)
        m["fail_bpm"] = not (bpm_min <= m["bpm"] <= bpm_max)
        m["fail_RR"] = bool(m["RR"] > 1 / 3)
        m["fail_sk"] = bool(m["sk"] < 2)
        m["cardiac"] = not (m["fail_bpm"] or m["fail_RR"] or m["fail_sk"])
        out.append(m)
    return out
