"""Dump the signal-processing chain, the rejection statistics, and the GLM design
for one window, as JSON for the cycle-rejection figure.

    .venv/bin/python Airflow/_scratch/dump_dsp.py MS18 eve 615 665 out.json

Read-only. Re-runs each stage of phase1_filter_downsample / detect_cycles /
build_continuous_series / fit_pooled_session_glm separately so every
intermediate can be plotted.
"""

import sys
import json

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, medfilt, resample

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config          # noqa: E402
from Airflow import airflow_glm as glm                # noqa: E402
from Airflow import airflow_qc as qc                  # noqa: E402
from probe_peak_gates import load_cache, session_cycles  # noqa: E402

METRIC = "RA"


def dec(t, y, n_out):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) <= n_out:
        idx = range(len(t))
    else:
        step = int(np.ceil(len(t) / n_out))
        idx = []
        for i in range(0, len(t), step):
            seg = y[i:i + step]
            if not len(seg) or not np.isfinite(seg).any():
                idx.append(i)
                continue
            idx.extend(sorted((i + int(np.nanargmin(seg)), i + int(np.nanargmax(seg)))))
    return [[round(float(t[i]), 3),
             (None if not np.isfinite(y[i]) else round(float(y[i]), 7))] for i in idx]


def crop(t, y, t0, t1, n_out=650):
    t = np.asarray(t, float)
    m = (t >= t0) & (t <= t1)
    return dec(t[m], np.asarray(y, float)[m], n_out)


def main():
    subj, sess_key = sys.argv[1], sys.argv[2]
    t0, t1 = float(sys.argv[3]), float(sys.argv[4])
    out_path = sys.argv[5]

    cache = load_cache()
    sess = cache[subj][sess_key]
    signal_raw = np.asarray(sess["signal_raw"], float)
    nsf = float(sess["sfreq"])
    t_raw = np.arange(len(signal_raw)) / nsf

    doc = {"subject": subj, "session": sess_key, "t0": t0, "t1": t1,
           "cache_sfreq": nsf, "target_sfreq": config.GLM_TARGET_SFREQ,
           "bandpass": list(config.GLM_CYCLE_BANDPASS),
           "final_hp": config.GLM_FINAL_HP[METRIC], "final_lp": config.GLM_FINAL_LP,
           "rf": list(config.GLM_RF_PARAMS[METRIC]), "metric": METRIC}

    # ── DSP chain, stage by stage (mirrors phase1_filter_downsample) ─────────
    x = signal_raw - np.nanmean(signal_raw)
    nyq = nsf / 2.0
    b_lp, a_lp = butter(1, config.GLM_CYCLE_BANDPASS[1] / nyq, btype="low")
    x_lp = filtfilt(b_lp, a_lp, x)
    b_hp, a_hp = butter(1, config.GLM_CYCLE_BANDPASS[0] / nyq, btype="high")
    x_hp = filtfilt(b_hp, a_hp, x_lp)
    sf = config.GLM_TARGET_SFREQ
    n_t = int(round(len(x_hp) * sf / nsf))
    x_ds = resample(x_hp, n_t)
    t_ds = np.arange(n_t) / sf
    med = float(np.nanmedian(x_ds))
    mad = float(np.nanmedian(np.abs(x_ds - med)))
    rsd = mad * 1.4826
    raw_z = (x_ds - med) / rsd

    win = int(round(config.GLM_MEDIAN_WIN_SEC * sf))
    win = win + 1 if win % 2 == 0 else win
    xm = medfilt(raw_z, kernel_size=max(win, 1))
    signs = np.sign(xm)
    signs[signs == 0] = 1
    cross = np.flatnonzero((signs[:-1] > 0) & (signs[1:] < 0)) + 1
    ibi = np.diff(cross)
    drop = np.zeros(len(cross), bool)
    drop[1:] = ibi < config.GLM_REFRACTORY_SEC * sf
    kept = cross[~drop]

    doc["dsp"] = {
        "raw": crop(t_raw, signal_raw, t0, t1),
        "centred": crop(t_raw, x, t0, t1),
        "lp": crop(t_raw, x_lp, t0, t1),
        "hp": crop(t_raw, x_hp, t0, t1),
        "z": crop(t_ds, raw_z, t0, t1),
        "medfilt": crop(t_ds, xm, t0, t1),
        "robust_sd": rsd, "median": med, "mad": mad,
        "crossings": [round(float(i / sf), 2) for i in cross if t0 <= i / sf <= t1],
        "onsets": [round(float(i / sf), 2) for i in kept if t0 <= i / sf <= t1],
        "n_cross": int(len(cross)), "n_onsets": int(len(kept)),
        "median_win_sec": config.GLM_MEDIAN_WIN_SEC,
        "refractory_sec": config.GLM_REFRACTORY_SEC,
    }

    # ── cycles + verdicts ────────────────────────────────────────────────────
    cycles, rejected, reason, report, _ = session_cycles(sess, subj, sess_key)
    rejected, reason = qc.apply_neighbour_rules(rejected, reason, config)
    med_RA = float(np.nanmedian([c["RA"] for c in cycles]))
    doc["median_RA"] = med_RA
    doc["cycles"] = [
        {"i": i, "onset": float(c["onset_time"]), "assign": float(c["assign_time"]),
         "RP": float(c["RP"]), "RA": float(c["RA"]), "RA_rel": float(c["RA"]) / med_RA,
         "n_peaks": int(c.get("n_peaks", -1)),
         "rejected": bool(rejected[i]), "reason": reason[i]}
        for i, c in enumerate(cycles)
        if c["onset_time"] < t1 and c["assign_time"] > t0]

    # ── series, before and after blanking ────────────────────────────────────
    n_samples = int(round(len(signal_raw) / nsf * sf))
    ser_keep, times_full = glm.build_continuous_series(
        cycles, n_samples, sf, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=None, rejected=rejected)
    ser_all, _ = glm.build_continuous_series(
        cycles, n_samples, sf, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
        missing_spans=None, rejected=np.zeros(len(cycles), bool))

    knot_t = np.array([c["assign_time"] for c in cycles])
    knot_v = np.array([c[METRIC] for c in cycles])
    o = np.argsort(knot_t)
    step_raw = np.interp(times_full, knot_t[o], knot_v[o],
                         left=knot_v[o][0], right=knot_v[o][-1])
    doc["series"] = {
        "knots": [[float(a), float(b)] for a, b in zip(knot_t[o], knot_v[o])
                  if t0 <= a <= t1],
        "interp": crop(times_full, step_raw, t0, t1),
        "filtered": crop(times_full, ser_all[METRIC], t0, t1),
        "blanked": crop(times_full, ser_keep[METRIC], t0, t1),
        "zscored": None,  # filled below
    }
    doc["blanked_spans"] = [[float(a), float(b)] for a, b in
                            glm.rejected_cycle_spans(cycles, rejected)
                            if a < t1 and b > t0]

    # ── trial admission ──────────────────────────────────────────────────────
    adm = glm.admit_trials(sess["trials_meta"], cycles, rejected, config)
    doc["trials"] = [
        {"index": int(m["trial_index"]), "span": [float(m["response_range_sec"][0]),
                                                  float(m["response_range_sec"][1])],
         "onset": float(m["code_time"]),
         "valid_frac": (None if not np.isfinite(adm[m["trial_index"]]["valid_frac"])
                        else float(adm[m["trial_index"]]["valid_frac"])),
         "n_cycles": int(adm[m["trial_index"]]["n_cycles_win"]),
         "admitted": bool(adm[m["trial_index"]]["admitted"])}
        for m in sess["trials_meta"]
        if m["response_range_sec"][0] < t1 and m["response_range_sec"][1] > t0]

    # ── GLM design, exactly as fit_pooled_session_glm builds it ──────────────
    tau, sigma = config.GLM_RF_PARAMS[METRIC]
    kernel_start, kernel_dur = -10.0, 30.0
    tk = np.arange(kernel_start, kernel_dur, 1.0 / sf)
    onset_off = int(round(-kernel_start * sf))
    crf = np.exp(-((tk - tau) ** 2) / (2 * sigma ** 2))
    dcrf = np.gradient(crf, tk)
    basis = glm.orthogonalize_and_normalize_basis(np.column_stack([crf, dcrf]))
    doc["basis"] = {
        "t": [round(float(v), 3) for v in tk[::2]],
        "crf_pre": [round(float(v), 5) for v in crf[::2]],
        "crf": [round(float(v), 5) for v in basis[:, 0][::2]],
        "dcrf": [round(float(v), 5) for v in basis[:, 1][::2]],
        "tau": tau, "sigma": sigma,
    }

    # z-score exactly as Pass A does: pooled over this subject's sessions
    subj_series = {}
    for sk2, ss2 in sorted(cache[subj].items()):
        if f"{subj}/{sk2}" in getattr(config, "SUBJECTS_EXCLUDE", {}):
            continue
        if sk2 == sess_key:
            subj_series[sk2] = ser_keep
            continue
        cy2, rj2, rs2, _, _ = session_cycles(ss2, subj, sk2)
        rj2, rs2 = qc.apply_neighbour_rules(rj2, rs2, config)
        n2 = int(round(len(ss2["signal_raw"]) / float(ss2["sfreq"]) * sf))
        s2, _ = glm.build_continuous_series(
            cy2, n2, sf, hp=config.GLM_FINAL_HP, lp=config.GLM_FINAL_LP,
            missing_spans=None, rejected=rj2)
        subj_series[sk2] = s2
    ser_z = glm.zscore_subject_series(subj_series, config)[sess_key]

    y = np.asarray(ser_z[METRIC], float)
    doc["series"]["zscored"] = crop(times_full, y, t0, t1)
    hp_m = config.GLM_FINAL_HP[METRIC]
    bh, ah = butter(1, hp_m / (sf / 2.0), btype="high")
    delta = np.zeros(n_samples)
    for m in sess["trials_meta"]:
        if adm[m["trial_index"]]["admitted"]:
            idx = int(round(float(m["code_time"]) * sf))
            if 0 <= idx < n_samples:
                delta[idx] = 1.0
    cols = []
    for bi in range(basis.shape[1]):
        conv = np.convolve(delta, basis[:, bi])[onset_off:onset_off + n_samples]
        conv = lfilter(bh, ah, conv)
        cols.append(conv - conv.mean())
    block = glm._spm_orth_columns(np.column_stack(cols))
    X = np.column_stack([np.ones(n_samples), block])
    valid = np.isfinite(y) & np.all(np.isfinite(block), axis=1)
    beta = np.linalg.pinv(X[valid]) @ y[valid]
    fitted = X @ beta
    resid = y - fitted
    ss_tot = float(np.nansum((y[valid] - np.nanmean(y[valid])) ** 2))
    ss_res = float(np.nansum(resid[valid] ** 2))

    doc["glm"] = {
        "impulses": [round(float(i / sf), 2) for i in np.flatnonzero(delta)
                     if t0 <= i / sf <= t1],
        "X_crf": crop(times_full, block[:, 0], t0, t1),
        "X_dcrf": crop(times_full, block[:, 1], t0, t1),
        "y": crop(times_full, y, t0, t1),
        "fitted": crop(times_full, np.where(valid, fitted, np.nan), t0, t1),
        "beta": [float(b) for b in beta],
        "n_valid": int(valid.sum()), "n_samples": int(n_samples),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "n_events": int(delta.sum()),
    }

    # ── cohort statistics behind the two session-relative thresholds ─────────
    excl = getattr(config, "SUBJECTS_EXCLUDE", {})
    logra, rps = [], []
    for s in sorted(cache):
        if s in excl:
            continue
        for sk, ss in sorted(cache[s].items()):
            if f"{s}/{sk}" in excl:
                continue
            cy, rj, rs, rep, _ = session_cycles(ss, s, sk)
            ras = np.array([c["RA"] for c in cy], float)
            good = np.isfinite(ras) & (ras > 0)
            if good.sum() < 5:
                continue
            m = np.median(np.log(ras[good]))
            tm = np.array([m2["response_range_sec"] for m2 in ss["trials_meta"]], float)
            w0, w1 = float(tm[:, 0].min()), float(tm[:, 1].max())
            for c, g in zip(cy, good):
                if not g or not (c["onset_time"] < w1 and c["assign_time"] > w0):
                    continue
                logra.append(float(np.log(c["RA"]) - m))
                rps.append(float(c["RP"]))
    logra = np.array(logra)
    rps = np.array(rps)
    cut = config.CYCLE_LOGRA_K * config.CYCLE_LOGRA_SCALE

    def hist(v, lo, hi, nb):
        h, e = np.histogram(v, bins=nb, range=(lo, hi))
        return {"counts": [int(c) for c in h], "lo": lo, "hi": hi, "n": int(len(v)),
                "n_over": int((v > hi).sum())}

    doc["stats"] = {
        "logra": hist(logra, -1.6, 1.9, 70),
        "logra_cut": float(cut),
        "logra_cut_x": float(np.exp(cut)),
        "logra_scale": config.CYCLE_LOGRA_SCALE,
        "logra_k": config.CYCLE_LOGRA_K,
        "logra_over": int((logra > cut).sum()),
        "logra_n": int(len(logra)),
        "rp": hist(rps, 0.0, 14.0, 70),
        "rp_cut": float(config.CYCLE_RP_MAX),
        "rp_over": int((rps > config.CYCLE_RP_MAX).sum()),
        "rp_pct": {p: float(np.percentile(rps, p)) for p in (50, 90, 95, 97, 99)},
        "window_logra": [float(np.log(c["RA"]) - np.log(med_RA)) for c in doc["cycles"]],
    }

    with open(out_path, "w") as f:
        json.dump(doc, f)
    print(f"{subj}/{sess_key}  crossings {doc['dsp']['n_cross']} -> onsets {doc['dsp']['n_onsets']}"
          f"  |  robust_sd {rsd:.6g}")
    print(f"  beta = {doc['glm']['beta']}  R2 {doc['glm']['r2']:.4f}  "
          f"valid {doc['glm']['n_valid']}/{n_samples}  events {doc['glm']['n_events']}")
    print(f"  cohort in-window: logRA n={len(logra)} over cut {doc['stats']['logra_over']} "
          f"({100*doc['stats']['logra_over']/len(logra):.2f}%)  cut = {np.exp(cut):.2f}x median")
    print(f"  RP n={len(rps)} over {config.CYCLE_RP_MAX}s: {doc['stats']['rp_over']} "
          f"({100*doc['stats']['rp_over']/len(rps):.2f}%)  pct {doc['stats']['rp_pct']}")


if __name__ == "__main__":
    main()
