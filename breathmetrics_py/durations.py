import numpy as np


def find_breath_durations(srate, inhale_onsets, inhale_offsets,
                          exhale_onsets, exhale_offsets,
                          inhale_pause_onsets, exhale_pause_onsets):
    inhale_onsets = np.asarray(inhale_onsets, dtype=float)
    exhale_onsets = np.asarray(exhale_onsets, dtype=float)
    inhale_offsets = np.asarray(inhale_offsets, dtype=float)
    exhale_offsets = np.asarray(exhale_offsets, dtype=float)
    inhale_pause_onsets = np.asarray(inhale_pause_onsets, dtype=float)
    exhale_pause_onsets = np.asarray(exhale_pause_onsets, dtype=float)

    inhale_durations = inhale_offsets - inhale_onsets
    exhale_durations = exhale_offsets - exhale_onsets

    inhale_pause_durations = np.full(inhale_pause_onsets.shape, np.nan)
    ok = ~np.isnan(inhale_pause_onsets)
    n = min(inhale_pause_onsets.size, exhale_onsets.size)
    idx = np.flatnonzero(ok[:n])
    inhale_pause_durations[idx] = exhale_onsets[idx] - inhale_pause_onsets[idx]

    exhale_pause_durations = np.full(exhale_pause_onsets.shape, np.nan)
    for e in range(exhale_pause_onsets.size):
        if not np.isnan(exhale_pause_onsets[e]) and e < inhale_onsets.size - 1:
            exhale_pause_durations[e] = inhale_onsets[e + 1] - exhale_pause_onsets[e]

    return (inhale_durations / srate, exhale_durations / srate,
            inhale_pause_durations / srate, exhale_pause_durations / srate)
