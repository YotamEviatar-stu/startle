"""Dump one session window (raw -> cycles -> gates -> series -> trials) as JSON
for the cycle-rejection figure.

    .venv/bin/python Airflow/_scratch/dump_window.py MS18 eve 615 665 out.json

Read-only. Mirrors run_glm_scoring Pass A via probe_peak_gates.session_cycles.
"""

import sys
import json

import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402
from probe_peak_gates import load_cache, session_cycles  # noqa: E402


def decimate_to(t, y, n_out):
    if len(t) <= n_out:
        return t, y
    step = int(np.ceil(len(t) / n_out))
    tt, yy = [], []
    for i in range(0, len(t), step):
        seg = y[i:i + step]
        ts = t[i:i + step]
        if not len(seg):
            continue
        finite = np.isfinite(seg)
        if not finite.any():
            tt.append(float(ts[0])); yy.append(None)
            continue
        jmin = int(np.nanargmin(seg)); jmax = int(np.nanargmax(seg))
        for j in sorted((jmin, jmax)):
            tt.append(float(ts[j])); yy.append(float(seg[j]))
    return np.array(tt), np.array(yy, dtype=object)


def main():
    subj, sess_key = sys.argv[1], sys.argv[2]
    t0, t1 = float(sys.argv[3]), float(sys.argv[4])
    out_path = sys.argv[5]

    cache = load_cache()
    sess = cache[subj][sess_key]

    cycles, rejected, reason, report, dur = session_cycles(sess, subj, sess_key)
    rejected, reason = qc.apply_neighbour_rules(rejected, reason, config)

    signal_raw = np.asarray(sess["signal_raw"], dtype=float)
    native_sfreq = float(sess["sfreq"])
    t_raw = np.arange(len(signal_raw)) / native_sfreq

    # raw_z exactly as the pipeline builds it
    if getattr(config, "QC_ENABLED", False):
        q = qc.qc_session(signal_raw, native_sfreq, config)
        sig_pipe = q["signal_clean"]
    else:
        sig_pipe = signal_raw
    raw_z, sfreq = glm.phase1_filter_downsample(
        sig_pipe, native_sfreq,
        target_sfreq=config.GLM_TARGET_SFREQ,
        bp_low=config.GLM_CYCLE_BANDPASS[0], bp_high=config.GLM_CYCLE_BANDPASS[1],
        zscore=config.GLM_ZSCORE_RAW,
        despike=(not getattr(config, "QC_ENABLED", False))
                and getattr(config, "GLM_DESPIKE_ENABLED", True),
        despike_k=getattr(config, "GLM_DESPIKE_K", 50.0),
        despike_max_run_sec=getattr(config, "GLM_DESPIKE_MAX_RUN_SEC", 1.0),
    )
    t_z = np.arange(len(raw_z)) / sfreq

    n_samples = int(round(len(signal_raw) / native_sfreq * sfreq))
    manual = [(float(a), float(b)) for a, b in
              getattr(config, "MANUAL_BAD_SPANS", {}).get(f"{subj}/{sess_key}", [])]

    ser_keep, _ = glm.build_continuous_series(
        cycles, n_samples, sfreq, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=manual, rejected=rejected)
    ser_all, _ = glm.build_continuous_series(
        cycles, n_samples, sfreq, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=manual, rejected=np.zeros(len(cycles), dtype=bool))
    t_ser = np.arange(n_samples) / sfreq

    blanked = glm.rejected_cycle_spans(cycles, rejected)
    admission = glm.admit_trials(sess["trials_meta"], cycles, rejected, config)

    med_RA = float(np.nanmedian([c["RA"] for c in cycles]))

    def win(a, b):
        return (a < t1) and (b > t0)

    cyc_out = []
    for i, c in enumerate(cycles):
        a, b = float(c["onset_time"]), float(c["assign_time"])
        if not win(a, b):
            continue
        cyc_out.append({
            "i": i, "onset": a, "assign": b, "RP": float(c["RP"]),
            "RA": float(c["RA"]), "RA_rel": float(c["RA"]) / med_RA,
            "n_peaks": int(c.get("n_peaks", -1)),
            "rejected": bool(rejected[i]), "reason": reason[i],
        })

    tr_out = []
    for meta in sess["trials_meta"]:
        w0, w1 = meta["response_range_sec"]
        if not win(float(w0), float(w1)):
            continue
        a = admission[meta["trial_index"]]
        tr_out.append({
            "index": int(meta["trial_index"]), "span": [float(w0), float(w1)],
            "valid_frac": None if not np.isfinite(a["valid_frac"]) else float(a["valid_frac"]),
            "n_cycles": int(a["n_cycles_win"]), "admitted": bool(a["admitted"]),
            "reason": a["admit_reason"],
        })

    def crop(t, y, n_out):
        m = (t >= t0) & (t <= t1)
        tt, yy = decimate_to(t[m], np.asarray(y, dtype=float)[m], n_out)
        return [[round(float(a), 3), (None if b is None or (isinstance(b, float) and not np.isfinite(b)) else round(float(b), 6))]
                for a, b in zip(tt, yy)]

    doc = {
        "subject": subj, "session": sess_key, "t0": t0, "t1": t1,
        "native_sfreq": native_sfreq, "sfreq": float(sfreq),
        "median_RA": med_RA,
        "signal_raw": crop(t_raw, signal_raw, 900),
        "raw_z": crop(t_z, raw_z, 900),
        "series_all": crop(t_ser, ser_all["RA"], 700),
        "series_keep": crop(t_ser, ser_keep["RA"], 700),
        "cycles": cyc_out,
        "blanked_spans": [[a, b] for a, b in blanked if win(a, b)],
        "trials": tr_out,
        "gate_order": list(getattr(qc, "GATE_ORDER", [])),
        "report": {k: v for k, v in report.items() if k in ("kept_frac", "kept_seconds_frac", "abstained", "n_ref_cycles")},
        "config": {
            "CYCLE_RP_MAX": config.CYCLE_RP_MAX,
            "CYCLE_LOGRA_K": config.CYCLE_LOGRA_K,
            "CYCLE_LOGRA_SCALE": config.CYCLE_LOGRA_SCALE,
            "CYCLE_LOSTLOCK_MIN_PEAKS": config.CYCLE_LOSTLOCK_MIN_PEAKS,
            "TRIAL_MIN_VALID_FRAC": config.TRIAL_MIN_VALID_FRAC,
        },
    }
    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"{subj}/{sess_key} {t0}-{t1}s: {len(cyc_out)} cycles, "
          f"{sum(c['rejected'] for c in cyc_out)} rejected, {len(tr_out)} trials, "
          f"blanked {doc['blanked_spans']}")
    for c in cyc_out:
        print(f"  cyc {c['i']:4d} {c['onset']:8.2f}-{c['assign']:8.2f} RP {c['RP']:5.2f} "
              f"RA {c['RA_rel']:6.2f}x  npk {c['n_peaks']}  "
              f"{'REJ ' + str(c['reason']) if c['rejected'] else 'keep'}")
    for tr in tr_out:
        print(f"  trial {tr['index']:3d} {tr['span'][0]:8.2f}-{tr['span'][1]:8.2f} "
              f"{tr['valid_frac']} n={tr['n_cycles']} "
              f"{'ADMIT' if tr['admitted'] else 'REJECT ' + str(tr['reason'])}")


if __name__ == "__main__":
    main()
