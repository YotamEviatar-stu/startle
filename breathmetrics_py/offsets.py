import numpy as np


def find_respiratory_offsets(resp, inhale_onsets, exhale_onsets,
                             inhale_pause_onsets, exhale_pause_onsets):
    resp = np.asarray(resp, dtype=float).ravel()
    inhale_onsets = np.asarray(inhale_onsets, dtype=int)
    exhale_onsets = np.asarray(exhale_onsets, dtype=int)

    inhale_offsets = np.full(inhale_onsets.shape, np.nan)
    exhale_offsets = np.full(exhale_onsets.shape, np.nan)

    for b in range(exhale_onsets.size):
        if np.isnan(inhale_pause_onsets[b]):
            inhale_offsets[b] = exhale_onsets[b] - 1
        else:
            inhale_offsets[b] = inhale_pause_onsets[b] - 1

    for b in range(exhale_onsets.size - 1):
        if np.isnan(exhale_pause_onsets[b]):
            exhale_offsets[b] = inhale_onsets[b + 1] - 1
        else:
            exhale_offsets[b] = exhale_pause_onsets[b] - 1

    final_window = resp[exhale_onsets[-1]:]
    above = np.flatnonzero(final_window > 0)
    lens = exhale_offsets[:-1] - exhale_onsets[:-1]
    avg_exhale_len = np.nanmean(lens) if lens.size else np.nan

    if above.size == 0 or np.isnan(avg_exhale_len):
        exhale_offsets[-1] = np.nan
    else:
        k = above[0] + 1
        if k < avg_exhale_len / 4.0 or k >= avg_exhale_len * 1.75:
            exhale_offsets[-1] = np.nan
        else:
            exhale_offsets[-1] = exhale_onsets[-1] + above[0]

    return inhale_offsets, exhale_offsets
