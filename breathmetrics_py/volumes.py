import numpy as np


def find_respiratory_volumes(resp, srate, inhale_onsets, exhale_onsets,
                             inhale_offsets, exhale_offsets):
    resp = np.asarray(resp, dtype=float).ravel()

    def _integrate(onsets, offsets):
        out = np.full(len(onsets), np.nan)
        for i in range(len(onsets)):
            off = offsets[i]
            on = onsets[i]
            if np.isnan(off) or np.isnan(on):
                continue
            a, b = int(on), int(off)
            if b < a:
                continue
            out[i] = np.abs(resp[a:b + 1]).sum()
        return (out / srate) * 1000.0

    inhale_volumes = _integrate(inhale_onsets, inhale_offsets)
    exhale_volumes = _integrate(exhale_onsets, exhale_offsets)
    return inhale_volumes, exhale_volumes
