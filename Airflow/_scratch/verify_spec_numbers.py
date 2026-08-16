"""Regression test: fixed reference numbers for MS18/eve, MS18/mor and RP06/mor,
regenerated from the Layer-1 cache. A FAIL means the pipeline's behaviour moved.

    /Users/yotameviatar/startle-1/.venv/bin/python Airflow/_scratch/verify_spec_numbers.py

No MFF reload, no writes, no config edits. Reads only
`{OUTPUT_DIR}/_cache/airflow_cache.pkl` and the pipeline code as it stands on disk.
Every line prints PASS or FAIL against the value written in the spec; a FAIL means either
the code changed or the spec is wrong, and both are worth knowing before a review.

The cycle numbering is the tagging view's (`airflow_qc_show.ipynb`, cell 11): 1-based
within the analysis window, after `flag_deep_breaths` has removed cycles. That is the
numbering used in every conversation about these sessions, so it is what the spec quotes.
"""

import os
import sys
import pickle

import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402

TOL = 0.005          # relative tolerance for ratios / fractions
N_FAIL = 0
_LEGACY_MAX_BREATH_RATIO = 5.0   # ex-config.QC_MAX_BREATH_RATIO; see session_state()


def check(label, got, want, tol=TOL, absolute=False):
    global N_FAIL
    if want is None:
        print(f"    ..   {label}: {got}")
        return got
    if isinstance(want, (bool, list, tuple, str)) or isinstance(got, (bool, list, tuple, str)):
        ok = got == want
    elif isinstance(want, (int, np.integer)) and isinstance(got, (int, np.integer)):
        ok = got == want
    else:
        d = abs(got - want)
        ok = d <= tol if absolute else d <= tol * max(abs(want), 1e-30)
    print(f"    {'PASS' if ok else 'FAIL'} {label}: got {got!r}  want {want!r}")
    if not ok:
        N_FAIL += 1
    return got


def load_cache():
    path = os.path.join(config.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)["sessions"]


def session_state(cache, subj, sess_key):
    """Exactly what airflow_qc_show.ipynb's compute_session_qc does, so cycle ids match."""
    sess = cache[subj][sess_key]
    signal_raw = np.asarray(sess["signal_raw"], dtype=float)
    native_sfreq = float(sess["sfreq"])
    trials_meta = sess["trials_meta"]

    q = qc.qc_session(signal_raw, native_sfreq, config)
    signal_clean = q["signal_clean"]

    raw_z, sfreq = glm.phase1_filter_downsample(
        signal_clean, native_sfreq,
        target_sfreq=config.GLM_TARGET_SFREQ,
        bp_low=config.GLM_CYCLE_BANDPASS[0], bp_high=config.GLM_CYCLE_BANDPASS[1],
        zscore=config.GLM_ZSCORE_RAW,
        despike=(not config.QC_ENABLED) and config.GLM_DESPIKE_ENABLED)

    cycles = glm.detect_cycles(
        raw_z, sfreq, signal_clean, native_sfreq,
        median_win_sec=config.GLM_MEDIAN_WIN_SEC,
        refractory_sec=config.GLM_REFRACTORY_SEC)

    missing = list(q["masked_spans"])
    # QC_MAX_BREATH_RATIO was deleted from airflow_config.py with the old trial
    # gates. This script reproduces spec §6's HISTORICAL numbers, which were
    # computed under flag_deep_breaths at 5.0x, so the value is pinned here.
    deep = qc.flag_deep_breaths(cycles, _LEGACY_MAX_BREATH_RATIO)
    deep_spans = [(c["onset_time"], c["assign_time"]) for c, b in zip(cycles, deep) if b]
    missing += deep_spans
    cycles = [c for c, b in zip(cycles, deep) if not b]

    d105 = [m["d105_time"] for m in trials_meta if m.get("d105_time") is not None]
    code = [m["code_time"] for m in trials_meta if m.get("code_time") is not None]
    win_start = min(d105) - config.GLM_PRE_FIXATION_SEC
    win_end = max(code) + config.GLM_POST_CODE_SEC

    n_samples = min(len(raw_z), int(np.ceil(win_end * sfreq)))
    if win_start > 0:
        missing = missing + [(0.0, win_start)]

    series, times = glm.build_continuous_series(
        cycles, n_samples, sfreq, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=missing)

    win_cycles = [c for c in cycles
                  if c["onset_time"] >= win_start and c["assign_time"] <= win_end]

    return dict(signal_raw=signal_raw, native_sfreq=native_sfreq, signal_clean=signal_clean,
                cycles=cycles, win_cycles=win_cycles, series=series, times=times,
                sfreq=sfreq, win_start=win_start, win_end=win_end,
                code_times=code, masked_spans=q["masked_spans"], deep_spans=deep_spans,
                breath_size=q["breath_size"],
                resp_ranges=[m["response_range_sec"] for m in trials_meta])


def ss_share(S, spans, metric="RA"):
    """Share of the analysis window's total sum of squares carried by `spans`."""
    t = S["times"]
    y = np.asarray(S["series"][metric], dtype=float)
    inwin = np.isfinite(y) & (t >= S["win_start"]) & (t < S["win_end"])
    yc = y - np.nanmean(y[inwin])
    total = float(np.nansum(yc[inwin] ** 2))
    m = np.zeros_like(inwin)
    for a, b in spans:
        m |= (t >= a) & (t < b)
    m &= inwin
    return float(np.nansum(yc[m] ** 2) / total), int(m.sum())


def response_epoch(S, onset, tol=0.6):
    """The trial epoch of spec §5: [code_n, code_{n+1}), keyed by the quoted onset."""
    hits = [(r0, r1) for r0, r1 in S["resp_ranges"] if abs(r0 - onset) <= tol]
    if len(hits) != 1:
        raise ValueError(f"no unique response epoch at onset {onset}: {hits}")
    return hits[0]


def overlapping(S, onset):
    r0, r1 = response_epoch(S, onset)
    return [(i, c) for i, c in enumerate(S["win_cycles"], 1)
            if c["assign_time"] > r0 and c["onset_time"] < r1]


# ══ MS18 / eve — the canonical example ═════════════════════════════════════════
def ms18_eve(cache):
    print("\nMS18 / eve  — spec §6 table and the 618-662 s walkthrough")
    S = session_state(cache, "MS18", "eve")
    cyc = S["win_cycles"]
    med = float(np.median([c["RA"] for c in cyc]))
    check("analysis window start (s)", round(S["win_start"], 1), 229.3)
    check("analysis window end (s)", round(S["win_end"], 1), 866.6)
    check("cycles in window", len(cyc), 167)
    check("median RA", med, 0.0029192)

    for cid, rp, ratio, knot in ((109, 3.30, 0.92, 633.10),
                                 (110, 5.30, 4.31, 638.40),
                                 (111, 5.40, 2.28, 643.80),
                                 (112, 3.70, 0.71, 647.50)):
        c = cyc[cid - 1]
        print(f"  cycle {cid}")
        check("RP (s)", round(c["RP"], 2), rp, tol=0.011, absolute=True)
        check("RA / median", round(c["RA"] / med, 2), ratio, tol=0.011, absolute=True)
        check("knot (assign_time, s)", round(c["assign_time"], 2), knot, tol=0.011, absolute=True)

    print("  blanking geometry")
    own = [(633.10, 643.80)]                 # the two invalid cycles' own spans
    bridged = [(633.10, 647.50)]             # last surviving knot -> next surviving knot
    f_own, n_own = ss_share(S, own)
    f_br, n_br = ss_share(S, bridged)
    check("own-spans samples", n_own, 107)
    check("own-spans share of sum-of-squares", round(f_own, 3), 0.315, tol=0.0015, absolute=True)
    check("bridged samples", n_br, 144)
    check("bridged share of sum-of-squares", round(f_br, 3), 0.324, tol=0.0015, absolute=True)

    print("  the four artifact breaths of this session")
    four = [(cyc[i - 1]["onset_time"], cyc[i - 1]["assign_time"]) for i in (18, 19, 110, 111)]
    f4, n4 = ss_share(S, four)
    t = S["times"]
    y = np.asarray(S["series"]["RA"], float)
    inwin = np.isfinite(y) & (t >= S["win_start"]) & (t < S["win_end"])
    check("seconds", round(n4 / S["sfreq"], 1), 20.9, tol=0.11, absolute=True)
    check("of analysed seconds", round(int(inwin.sum()) / S["sfreq"], 1), 637.3, tol=0.11, absolute=True)
    check("fraction of samples", round(n4 / int(inwin.sum()), 4), 0.0328, tol=0.0006, absolute=True)
    check("share of sum-of-squares", round(f4, 3), 0.611, tol=0.0015, absolute=True)

    print("  are they finite in y as the code stands today?")
    for cid in (18, 19, 110, 111):
        c = cyc[cid - 1]
        m = (t >= c["onset_time"]) & (t < c["assign_time"])
        check(f"cycle {cid} finite fraction in series['RA']", float(np.mean(np.isfinite(y[m]))), 1.0)

    print("  walkthrough window 618-662 s, after blanking [633.1, 647.5)")
    keep = [c for c in S["cycles"]
            if not any(abs(c["assign_time"] - cyc[i - 1]["assign_time"]) < 1e-6 for i in (110, 111))]
    s2, _ = glm.build_continuous_series(
        keep, len(y), S["sfreq"], hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=list(S["masked_spans"]) + [(0.0, S["win_start"]), (633.10, 647.50)])
    y2 = np.asarray(s2["RA"], float)
    w = (t >= 618.0) & (t <= 662.0)
    n_rows, n_nan = int(w.sum()), int(np.sum(~np.isfinite(y2[w])))
    check("rows in window", n_rows, 441)
    check("rows carrying NaN", n_nan, 144)
    check("rows into the solve", n_rows - n_nan, 297)

    print("  trial admission over [code_n, code_n+1), floor 0.60")
    bad_ids = {110, 111}
    for onset, want_end, want_ok, want_n in ((620.4, 639.14, 4, 6), (639.1, 656.22, 4, 5)):
        r0, r1 = response_epoch(S, onset)
        ov = overlapping(S, onset)
        n_ok = sum(1 for i, _ in ov if i not in bad_ids)
        check(f"trial at {onset} s — epoch end", round(r1, 2), want_end, tol=0.011, absolute=True)
        check(f"trial at {onset} s — valid breaths", n_ok, want_ok)
        check(f"trial at {onset} s — breaths in epoch", len(ov), want_n)
        check(f"trial at {onset} s — admitted", bool(n_ok / len(ov) >= 0.60), True)


# ══ RP06 / mor — the gate that cannot see it ═══════════════════════════════════
def rp06_mor(cache):
    print("\nRP06 / mor  — spec §2 defect 2 and §6")
    S = session_state(cache, "RP06", "mor")
    x, sf = S["signal_raw"], S["native_sfreq"]
    t = np.arange(len(x)) / sf
    check("analysis window start (s)", round(S["win_start"], 1), 1036.0)
    check("analysis window end (s)", round(S["win_end"], 1), 1713.0)
    check("cycles in window", len(S["win_cycles"]), 255)

    A = qc.session_breath_size(x, sf)
    check("session_breath_size A", A, 0.000219662)
    sub = (t >= 1330.0) & (t <= 1356.0)
    peak = float(np.max(x[sub]))
    check("peak value in 1330-1356 s", peak, 0.00327984)
    check("peak time (s)", round(float(t[sub][int(np.argmax(x[sub]))]), 2), 1340.40, tol=0.011, absolute=True)
    check("peak in units of A", round(peak / A, 2), 14.93, tol=0.011, absolute=True)

    # §2 defect 2 (flag_excursions blind to multi-second artifacts) is CLOSED:
    # the sample-level gate was deleted 2026-08-15, so there is nothing left to
    # be blind. Reproduced here only as the invariant that qc_session repairs
    # and masks nothing at all.
    print("  sample gate deleted — qc_session repairs and masks nothing")
    check("spans masked by qc_session", len(S["masked_spans"]), 0)
    check("signal_clean is signal_raw", bool(np.array_equal(S["signal_clean"], S["signal_raw"])), True)

    print("  what flag_deep_breaths caught, and what it left behind")
    check("deep-breath spans dropped", len(S["deep_spans"]), 3)
    check("the artifact span is one of them", (1337.4, 1341.2) in [(round(a, 1), round(b, 1)) for a, b in S["deep_spans"]], True)
    nxt = [c for c in S["win_cycles"] if abs(c["onset_time"] - 1341.20) < 0.05]
    check("a recovery cycle starts at 1341.20 s", len(nxt), 1)
    if nxt:
        med = float(np.median([c["RA"] for c in S["win_cycles"]]))
        check("recovery RP (s)", round(nxt[0]["RP"], 2), 9.50, tol=0.011, absolute=True)
        check("recovery RA / median", round(nxt[0]["RA"] / med, 2), 2.10, tol=0.011, absolute=True)
        f, n = ss_share(S, [(nxt[0]["onset_time"], nxt[0]["assign_time"])])
        check("recovery share of sum-of-squares", round(f, 3), 0.214, tol=0.0015, absolute=True)


# ══ MS18 / mor — the sensor event and the gap-fill rule ════════════════════════
def ms18_mor(cache):
    print("\nMS18 / mor  — spec §6, gap-fill and the rejected trial")
    S = session_state(cache, "MS18", "mor")
    cyc = S["win_cycles"]
    med = float(np.median([c["RA"] for c in cyc]))
    check("cycles in window", len(cyc), 129)
    check("median RA", med, 0.0025244)

    spans = [(cyc[i - 1]["onset_time"], cyc[i - 1]["assign_time"]) for i in range(95, 108)]
    f, n = ss_share(S, spans)
    t = S["times"]
    y = np.asarray(S["series"]["RA"], float)
    inwin = np.isfinite(y) & (t >= S["win_start"]) & (t < S["win_end"])
    check("cycles 95-107 seconds", round(n / S["sfreq"], 1), 85.4, tol=0.11, absolute=True)
    check("of analysed seconds", round(int(inwin.sum()) / S["sfreq"], 1), 671.1, tol=0.11, absolute=True)
    check("fraction of samples", round(n / int(inwin.sum()), 4), 0.1273, tol=0.0006, absolute=True)
    check("share of sum-of-squares", round(f, 3), 0.246, tol=0.0015, absolute=True)

    print("  gap-fill on the illustrative gate (RP >= 7 s or RA <= 0.10x median)")
    ids = list(range(95, 108))
    gate = [bool(cyc[i - 1]["RP"] >= 7.0 or cyc[i - 1]["RA"] / med <= 0.10) for i in ids]
    filled = list(gate)
    for k in range(1, len(gate) - 1):
        if gate[k - 1] and gate[k + 1]:
            filled[k] = True
    flipped = [ids[k] for k in range(len(ids)) if filled[k] and not gate[k]]
    check("cycles flipped by gap-fill", flipped, [101, 103])

    print("  trial 29 (onset 662.7 s) — the one place the admission branch fires")
    r0, r1 = response_epoch(S, 662.7)
    ov = overlapping(S, 662.7)
    bad = {i for i, c in ov if c["RP"] >= 7.0 or c["RA"] / med <= 0.10}
    n_ok = sum(1 for i, _ in ov if i not in bad)
    check("epoch end", round(r1, 2), 680.91, tol=0.011, absolute=True)
    check("breaths in epoch", len(ov), 3)
    check("valid breaths", n_ok, 1)
    check("admitted at floor 0.60", bool(n_ok / len(ov) >= 0.60), False)


if __name__ == "__main__":
    cache = load_cache()
    ms18_eve(cache)
    rp06_mor(cache)
    ms18_mor(cache)
    print(f"\n{'ALL CHECKS PASSED' if N_FAIL == 0 else f'{N_FAIL} CHECK(S) FAILED'}")
    sys.exit(1 if N_FAIL else 0)
