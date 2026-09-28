import numpy as np


def create_respiratory_erp_matrix(resp, event_array, pre_samples, post_samples,
                                  append_nans=False):
    resp = np.asarray(resp, dtype=float).ravel()
    n = resp.size
    width = int(pre_samples) + int(post_samples) + 1

    rows = []
    trial_events = []
    rejected_events = []
    trial_inds = []
    rejected_inds = []

    for i, ev in enumerate(np.asarray(event_array, dtype=float)):
        if np.isnan(ev):
            rejected_events.append(ev)
            rejected_inds.append(i)
            continue
        ev = int(ev)
        idx = np.arange(ev - int(pre_samples), ev + int(post_samples) + 1)
        inside = (idx >= 0) & (idx < n)

        if not append_nans:
            if inside.all():
                rows.append(resp[idx])
                trial_events.append(ev)
                trial_inds.append(i)
            else:
                rejected_events.append(ev)
                rejected_inds.append(i)
        else:
            row = np.full(width, np.nan)
            row[inside] = resp[idx[inside]]
            if np.all(np.isnan(row)):
                rejected_events.append(ev)
                rejected_inds.append(i)
            else:
                rows.append(row)
                trial_events.append(ev)
                trial_inds.append(i)

    matrix = np.vstack(rows) if rows else np.empty((0, width))
    return (matrix, np.asarray(trial_events), np.asarray(rejected_events),
            np.asarray(trial_inds, dtype=int), np.asarray(rejected_inds, dtype=int))
