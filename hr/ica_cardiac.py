import os
import re
import sys
from datetime import datetime

import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing.ctps_ import ctps
import neurokit2 as nk
import pandas as pd
from defusedxml.minidom import parse
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.stats import skew
from scipy.io import savemat, loadmat
import subprocess
import tempfile
from autoreject import get_rejection_threshold
from pyprep.find_noisy_channels import NoisyChannels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr import config as cfg

MATLAB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matlab")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out")

FIT_BAND = (1.0, 40.0)
FIT_DECIM = 4
N_COMPONENTS_MAX = 64
SAMPLES_PER_COMP_SQ = 30
REJECT_EPOCH_S = 1.0
ICA_METHOD = "infomax"
ICA_FIT_PARAMS = {"extended": True}
RANDOM_STATE = 97
APPLY_HP_HZ = 0.1

PREP_RANSAC = False


IBI_BPM_RANGE = (40.0, 180.0)
IBI_REL_DEV = 0.25
IBI_MED_N = 11
MAX_GAP_S = 3.0

CTPS_BAND = (8.0, 16.0)
CTPS_TMIN, CTPS_TMAX = -0.5, 0.5
CTPS_THRESHOLD = 0.3

CARACAS_SRATE = 250.0

ECG_METHOD = "neurokit"

PAIR_WIN_S = 0.4
AGREE_MIN_MATCHED = 0.95
AGREE_MAX_DIBI_MS = 25.0

N_AVG_BEATS = 4
SPO2_LAG_S = 3

PULSE_CHANNEL = "SpO2-Pulse"
PULSE_VALID_BPM = (20.0, 250.0)


def session_key(mff_path):
    path = os.path.normpath(mff_path)
    parts = os.path.basename(path).split("_")
    parent = os.path.dirname(path)
    folder = os.path.basename(os.path.dirname(parent)) if os.path.basename(parent) == "EEG" else parts[0]
    return cfg.SUBJECT_ID.get(folder, folder), parts[1].lower()


def subject_folder(subj):
    return {v: k for k, v in cfg.SUBJECT_ID.items()}.get(subj, subj)


def excluded(subj, sess):
    return subj in cfg.SUBJECTS_EXCLUDE or f"{subj}/{sess}" in cfg.SUBJECTS_EXCLUDE


def find_mff_file(eeg_folder, sess):
    hits = sorted(n for n in os.listdir(eeg_folder)
                  if f"_{sess}_" in n.lower() and n.lower().endswith(".mff") and not n.startswith("._"))
    done = [n for n in hits if os.path.exists(os.path.join(eeg_folder, n, "epochs.xml"))]
    return os.path.join(eeg_folder, done[0]) if done else None


def din_times(mff_path):
    t_start = datetime.fromisoformat(
        parse(os.path.join(mff_path, "info.xml")).getElementsByTagName("recordTime")[0].firstChild.data)
    out = []
    for name in sorted(os.listdir(mff_path)):
        if not (name.startswith("Events_") and name.endswith(".xml")):
            continue
        for code, begin in read_events(os.path.join(mff_path, name)):
            if code.startswith("D"):
                out.append((code, (datetime.fromisoformat(begin) - t_start).total_seconds()))
    return sorted(out, key=lambda e: e[1])


def read_events(xml_path):
    try:
        events = parse(xml_path).getElementsByTagName("event")
        pairs = []
        for ev in events:
            code = ev.getElementsByTagName("code")
            begin = ev.getElementsByTagName("beginTime")
            if code and code[0].firstChild is not None and begin:
                pairs.append((code[0].firstChild.data, begin[0].firstChild.data))
        return pairs
    except Exception:
        text = open(xml_path, encoding="utf-8", errors="replace").read()
        pairs = []
        for block in re.findall(r"<event>(.*?)</event>", text, flags=re.S):
            code = re.search(r"<code>([^<]*)</code>", block)
            begin = re.search(r"<beginTime>([^<]*)</beginTime>", block)
            if code and begin:
                pairs.append((code.group(1), begin.group(1)))
        return pairs


def session_spans(mff_path, key):
    ev = din_times(mff_path)
    d101 = [t for c, t in ev if c == "D101"]
    if not d101:
        raise RuntimeError(f"{key}: no D101")
    d124 = [t for c, t in ev if c == "D124" and t > d101[0]]
    end = d124[0] if d124 else recording_end(mff_path)
    d102 = [t for c, t in ev if c == "D102"]
    pairs = list(zip(d102[0::2], d102[1::2]))
    auto = {"pre": next((p for p in pairs if p[1] < d101[0]), None),
            "post": next((p for p in pairs if p[0] > end), None) if d124 else None}
    spans = {**auto, **cfg.MANUAL_REST_SPANS.get(key, {})}
    lo = spans["pre"][0] if spans["pre"] else d101[0]
    hi = spans["post"][1] if spans["post"] else end
    return {"full": (lo, hi), "task": (d101[0], end), "d124_found": bool(d124),
            "rest_pre": spans["pre"], "rest_post": spans["post"]}


def recording_end(mff_path):
    ep = parse(os.path.join(mff_path, "epochs.xml")).getElementsByTagName("epoch")
    begin = int(ep[0].getElementsByTagName("beginTime")[0].firstChild.data)
    end = int(ep[-1].getElementsByTagName("endTime")[0].firstChild.data)
    return (end - begin) / 1e6


def pulse_grid(raw, t0, t1):
    x = raw.get_data(picks=[PULSE_CHANNEL])[0]
    sr = raw.info["sfreq"]
    grid = np.arange(np.ceil(t0), np.floor(t1) + 1.0)
    idx = np.clip(np.round((grid - t0) * sr).astype(int), 0, x.size - 1)
    bpm = x[idx].astype(float)
    bpm[(bpm < PULSE_VALID_BPM[0]) | (bpm > PULSE_VALID_BPM[1])] = np.nan
    return grid, bpm


def readable_mff(mff_path):
    broken = []
    for name in os.listdir(mff_path):
        if name.startswith("Events_") and name.endswith(".xml"):
            try:
                parse(os.path.join(mff_path, name))
            except Exception:
                broken.append(name)
    if not broken:
        return mff_path
    fixed = os.path.join(OUT_DIR, "_repaired", os.path.basename(os.path.normpath(mff_path)))
    os.makedirs(fixed, exist_ok=True)
    for name in os.listdir(mff_path):
        if name.startswith("._"):
            continue
        dst = os.path.join(fixed, name)
        if name in broken:
            text = open(os.path.join(mff_path, name), encoding="utf-8", errors="replace").read()
            close = text.find("</eventTrack>")
            if close >= 0:
                text = text[:close + len("</eventTrack>")] + "\n"
            else:
                text = text[:text.rfind("</event>") + len("</event>")] + "\n</eventTrack>\n"
            open(dst, "w", encoding="utf-8").write(text)
        elif not os.path.lexists(dst):
            os.symlink(os.path.join(mff_path, name), dst)
    return fixed


def load_eeg(mff_path, t0, t1, bad_spans, with_pulse=False):
    raw = mne.io.read_raw_egi(readable_mff(mff_path), preload=False, verbose="error")
    raw.pick(mne.pick_types(raw.info, eeg=True) .tolist()
             + ([raw.ch_names.index(PULSE_CHANNEL)] if with_pulse and PULSE_CHANNEL in raw.ch_names else []))
    if bad_spans:
        onset = [a for a, _ in bad_spans]
        dur = [b - a for a, b in bad_spans]
        raw.set_annotations(raw.annotations + mne.Annotations(
            onset, dur, ["BAD_manual"] * len(onset), orig_time=raw.annotations.orig_time))
    raw.crop(tmin=t0, tmax=t1)
    raw.load_data(verbose="error")
    return raw


def find_bad_channels(fit_raw):
    data = fit_raw.get_data(reject_by_annotation="omit")
    info = fit_raw.info.copy()
    info["bads"] = []
    nc = NoisyChannels(mne.io.RawArray(data, info, verbose="error"), do_detrend=False,
                       random_state=RANDOM_STATE)
    nc.find_all_bads(ransac=PREP_RANSAC)
    return sorted(set(nc.get_bads()))


def fit_rejection(fit_raw):
    epochs = mne.make_fixed_length_epochs(fit_raw, duration=REJECT_EPOCH_S, preload=False,
                                          reject_by_annotation=True, verbose="error")
    epochs.decimate(FIT_DECIM, verbose="error")
    epochs.load_data()
    reject = get_rejection_threshold(epochs, random_state=RANDOM_STATE, ch_types="eeg", verbose=False)
    n_total = len(epochs)
    epochs.drop_bad(reject=reject, verbose="error")
    return reject, n_total, len(epochs), len(epochs) * epochs.times.size


def n_components_for(n_samples, n_channels):
    by_data = int(np.floor(np.sqrt(n_samples / SAMPLES_PER_COMP_SQ)))
    return max(2, min(N_COMPONENTS_MAX, by_data, n_channels - 1))


def sasica_version():
    return " ".join(
        subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        for d in (cfg.SASICA_DIR, cfg.HEART_FUNCTIONS_DIR))


def caracas_official(src, sr):
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.mat"), os.path.join(tmp, "out.mat")
        savemat(fin, {"src": src, "fs": float(sr)})
        paths = [cfg.FIELDTRIP_DIR, os.path.join(cfg.FIELDTRIP_DIR, "external", "stats"), MATLAB_DIR,
                 os.path.join(cfg.SASICA_DIR, "CARACAS"), cfg.HEART_FUNCTIONS_DIR]
        cmd = ("addpath('%s'); ft_defaults; addpath('%s', '%s', '%s', '%s', '-begin'); caracas_run('%s', '%s');"
               % (paths[0], *paths[1:], fin, fout))
        r = subprocess.run([cfg.MATLAB_BIN, "-batch", cmd], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(fout):
            raise RuntimeError("CARACAS.m failed: " + r.stdout[-2000:] + r.stderr[-2000:])
        m = loadmat(fout)
    heart = [int(c) for c in np.atleast_1d(m["heart_IC"].ravel())]
    meas = {k: np.asarray(m[k], float).ravel() for k in ("bpm", "sk", "ku", "rr_cv", "rampl_cv")}
    return heart, meas


def good_mask(raw, n_out, sr_out):
    mask = np.ones(n_out, bool)
    for ann in raw.annotations:
        if not ann["description"].upper().startswith("BAD"):
            continue
        a = ann["onset"] - raw.first_time
        lo = max(int(np.floor(a * sr_out)), 0)
        hi = min(int(np.ceil((a + ann["duration"]) * sr_out)), n_out)
        mask[lo:hi] = False
    return mask


def ecg_beats(x, sr):
    clean = nk.ecg_clean(x, sampling_rate=sr, method=ECG_METHOD)
    _, info = nk.ecg_peaks(clean, sampling_rate=sr, method=ECG_METHOD, correct_artifacts=True)
    idx = np.asarray(info["ECG_R_Peaks"], dtype=int)
    grade = nk.ecg_quality(clean, rpeaks=idx, sampling_rate=sr, method="zhao2018", approach="fuzzy")
    per_beat = nk.ecg_quality(clean, rpeaks=idx, sampling_rate=sr, method="averageQRS")[idx]
    return idx, grade, np.asarray(per_beat, dtype=float)


def beat_agreement(a, b):
    if a.size < 3 or b.size < 3:
        return {"matched": 0.0, "delay_ms": np.nan, "delay_sd_ms": np.nan,
                "dibi_median_ms": np.nan, "dibi_p95_ms": np.nan}
    j = np.clip(np.searchsorted(b, a), 1, b.size - 1)
    nn = np.where(np.abs(b[j] - a) < np.abs(b[j - 1] - a), j, j - 1)
    dly = b[nn] - a
    ok = np.abs(dly) < PAIR_WIN_S
    k = np.flatnonzero(ok[:-1] & ok[1:] & (np.diff(nn) == 1))
    dibi = np.abs((a[k + 1] - a[k]) - (b[nn[k + 1]] - b[nn[k]])) * 1000.0
    return {"matched": float(ok.mean()),
            "delay_ms": float(np.median(dly[ok]) * 1000.0) if ok.any() else np.nan,
            "delay_sd_ms": float(np.std(dly[ok]) * 1000.0) if ok.any() else np.nan,
            "dibi_median_ms": float(np.median(dibi)) if dibi.size else np.nan,
            "dibi_p95_ms": float(np.percentile(dibi, 95)) if dibi.size else np.nan}


def four_beat_grid(beat_t, ok, grid):
    ibi = np.diff(beat_t)
    b4 = np.full(ibi.size, np.nan)
    for i in range(N_AVG_BEATS - 1, ibi.size):
        if ok[i - N_AVG_BEATS + 1:i + 1].all():
            b4[i] = 60.0 / ibi[i - N_AVG_BEATS + 1:i + 1].mean()
    out = np.full(grid.size, np.nan)
    j = np.searchsorted(beat_t[1:], grid, side="right") - 1
    out[j >= 0] = b4[j[j >= 0]]
    return out


def vs_pulse(bpm4, pulse, lag):
    ref = np.full(pulse.size, np.nan)
    ref[:pulse.size - lag] = pulse[lag:] if lag else pulse
    m = np.isfinite(bpm4) & np.isfinite(ref)
    if m.sum() < 3:
        return np.nan, np.nan, int(m.sum())
    return (float(np.corrcoef(bpm4[m], ref[m])[0, 1]),
            float(np.median(np.abs(bpm4[m] - ref[m]))), int(m.sum()))


def rmssd_ms(beat_t, ok):
    ibi = np.diff(beat_t)
    k = np.flatnonzero(ok[:-1] & ok[1:])
    return float(np.sqrt(np.mean((ibi[k + 1] - ibi[k]) ** 2)) * 1000.0) if k.size else np.nan


def clean_ibis(beat_t):
    ibi = np.diff(beat_t)
    t_mid = beat_t[:-1] + ibi / 2.0
    ok = (ibi >= 60.0 / IBI_BPM_RANGE[1]) & (ibi <= 60.0 / IBI_BPM_RANGE[0])
    half = IBI_MED_N // 2
    ref = np.where(ok, ibi, np.nan)
    local = np.array([np.nanmedian(ref[max(i - half, 0):i + half + 1])
                      if np.isfinite(ref[max(i - half, 0):i + half + 1]).any() else np.nan
                      for i in range(ibi.size)])
    ok &= np.abs(ibi - local) <= IBI_REL_DEV * local
    return t_mid, 60.0 / ibi, ok


def to_grid(t_mid, bpm, ok, grid):
    t, v = t_mid[ok], bpm[ok]
    out = np.full(grid.size, np.nan)
    if t.size < 2:
        return out
    j = np.searchsorted(t, grid)
    inside = (j > 0) & (j < t.size)
    jj = j[inside]
    gap = t[jj] - t[jj - 1]
    val = np.interp(grid[inside], t, v)
    val[gap > MAX_GAP_S] = np.nan
    out[inside] = val
    return out


def ctps_components(ica, src1000, sr, beats1000, good1000):
    sos = butter(4, CTPS_BAND, "bandpass", fs=sr, output="sos")
    filt = sosfiltfilt(sos, src1000, axis=1)
    lo, hi = int(round(CTPS_TMIN * sr)), int(round(CTPS_TMAX * sr))
    keep = [b for b in beats1000
            if b + lo >= 0 and b + hi < filt.shape[1] and good1000[b + lo:b + hi + 1].all()]
    epochs = np.stack([filt[:, b + lo:b + hi + 1] for b in keep])
    _, p_vals, _ = ctps(epochs)
    scores = p_vals.max(-1)
    idx = np.flatnonzero(scores >= CTPS_THRESHOLD)
    return idx[np.argsort(scores[idx])[::-1]], scores, CTPS_THRESHOLD, len(keep)


def run_session(mff_path):
    subj, sess = session_key(mff_path)
    key = f"{subj}/{sess}"
    spans = session_spans(mff_path, key)
    t0, t1 = spans["full"]
    raw = load_eeg(mff_path, t0, t1, cfg.MANUAL_BAD_SPANS.get(key, []), with_pulse=True)
    if PULSE_CHANNEL not in raw.ch_names:
        raise RuntimeError(f"{key}: no {PULSE_CHANNEL} channel")
    grid, pulse = pulse_grid(raw, t0, t1)
    raw.pick("eeg")
    fit_raw = raw.copy().filter(*FIT_BAND, verbose="error")
    del raw

    bads = find_bad_channels(fit_raw)
    fit_raw.info["bads"] = bads
    fit_raw.set_eeg_reference("average", verbose="error")
    reject, n_ep, n_ep_kept, n_fit_samples = fit_rejection(fit_raw)
    n_good_ch = len(fit_raw.ch_names) - len(bads)
    n_comp = n_components_for(n_fit_samples, n_good_ch)

    ica = ICA(n_components=n_comp, method=ICA_METHOD, fit_params=ICA_FIT_PARAMS,
              random_state=RANDOM_STATE, max_iter="auto")
    ica.fit(fit_raw, decim=FIT_DECIM, reject=reject, tstep=REJECT_EPOCH_S,
            reject_by_annotation=True, verbose="error")

    sr = fit_raw.info["sfreq"]
    src1000 = ica.get_sources(fit_raw).get_data()
    good1000 = good_mask(fit_raw, src1000.shape[1], sr)
    del fit_raw
    step = int(round(sr / CARACAS_SRATE))
    good = good1000[::step]
    src = src1000[:, ::step] * good

    caracas, meas = caracas_official(src, CARACAS_SRATE)
    caracas = sorted(caracas, key=lambda c: -meas["sk"][c])
    sign = {c: (1.0 if skew(src[c]) >= 0 else -1.0) for c in caracas}

    def spike_times(c):
        idx, grade, q = ecg_beats(src1000[c] * sign[c], sr)
        keep = good1000[idx]
        return t0 + idx[keep] / sr, idx[keep], grade, q[keep]

    if not caracas:
        stats = {"session": key, "hr_source": "none: no CARACAS cardiac component", "caracas_version": sasica_version(),
                 "n_caracas": 0, "n_components": int(ica.n_components_), "n_bads": len(bads),
                 "reject_uv": float(reject["eeg"] * 1e6), "fit_epochs_kept": f"{n_ep_kept}/{n_ep}",
                 "window_s": float(t1 - t0)}
        os.makedirs(OUT_DIR, exist_ok=True)
        stem = os.path.join(OUT_DIR, f"{subj}_{sess}")
        ica.save(stem + "-ica.fif", overwrite=True, verbose="error")
        np.savez_compressed(stem + ".npz", t_1hz=grid, pulse_bpm_1hz=pulse, caracas=np.array([], int),
                            hr_source=stats["hr_source"],
                            bads=np.array(bads), analysis_window=np.array([t0, t1]))
        return stats

    primary = caracas[0]
    beat_t, beat_idx, quality_grade, beat_quality = spike_times(primary)
    hr_source = f"IC{primary} (CARACAS)"
    if len(caracas) > 1:
        other_t = spike_times(caracas[1])[0]
        check_with = f"IC{caracas[1]} (2nd CARACAS)"
    else:
        other_t, check_with = np.array([]), "none"
    agree = beat_agreement(beat_t, other_t)
    cross_ok = bool(agree["matched"] >= AGREE_MIN_MATCHED and agree["dibi_median_ms"] <= AGREE_MAX_DIBI_MS)

    ctps_idx, ctps_scores, ctps_thr, n_epochs = ctps_components(ica, src1000, sr, beat_idx, good1000)
    cardiac = sorted(set(caracas) | set(int(c) for c in ctps_idx))
    ica.exclude = cardiac

    t_mid, bpm, ok = clean_ibis(beat_t)
    bpm_1hz = to_grid(t_mid, bpm, ok, grid)
    bpm4 = four_beat_grid(beat_t, ok, grid)
    r0, d0, _ = vs_pulse(bpm4, pulse, 0)
    rl, dl, n_cmp = vs_pulse(bpm4, pulse, SPO2_LAG_S)

    stats = {
        "session": key, "hr_source": hr_source, "caracas_version": sasica_version(), "n_caracas": len(caracas),
        "caracas": " ".join(map(str, caracas)),
        "caracas_sk": meas["sk"][primary], "caracas_ku": meas["ku"][primary],
        "caracas_rr_cv": meas["rr_cv"][primary], "caracas_bpm": meas["bpm"][primary],
        "nk_quality_zhao2018": quality_grade,
        "beat_quality_median": float(np.median(beat_quality)) if beat_quality.size else np.nan,
        "n_beats": int(beat_t.size), "n_ibi_rejected": int((~ok).sum()),
        "pct_nan_1hz": float(100.0 * np.mean(~np.isfinite(bpm_1hz))),
        "rmssd_ms": rmssd_ms(beat_t, ok),
        "r_pulse_lag0": r0, "mad_pulse_lag0": d0,
        f"r_pulse_lag{SPO2_LAG_S}": rl, f"mad_pulse_lag{SPO2_LAG_S}": dl, "n_pulse_compared": n_cmp,
        "cross_check_with": check_with, "cross_ok": cross_ok,
        "cross_matched": agree["matched"], "cross_delay_ms": agree["delay_ms"],
        "cross_dibi_median_ms": agree["dibi_median_ms"], "cross_dibi_p95_ms": agree["dibi_p95_ms"],
        "removed": " ".join(map(str, cardiac)), "n_ctps_epochs": n_epochs,
        "n_components": int(ica.n_components_), "n_bads": len(bads),
        "reject_uv": float(reject["eeg"] * 1e6), "fit_epochs_kept": f"{n_ep_kept}/{n_ep}",
        "window_s": float(t1 - t0), "d124_found": spans["d124_found"],
        "rest_pre_found": spans["rest_pre"] is not None, "rest_post_found": spans["rest_post"] is not None,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.join(OUT_DIR, f"{subj}_{sess}")
    ica.save(stem + "-ica.fif", overwrite=True, verbose="error")
    np.savez_compressed(
        stem + ".npz", t_1hz=grid, bpm_1hz=bpm_1hz, bpm_4beat_1hz=bpm4, pulse_bpm_1hz=pulse,
        beat_times_s=beat_t, beat_quality=beat_quality, ibi_t_s=t_mid, ibi_bpm=bpm, ibi_ok=ok, hr_source=hr_source,
        other_beat_times_s=other_t, cardiac=np.array(cardiac),
        caracas=np.array(caracas), primary=primary,
        ctps_scores=ctps_scores, ctps_threshold=ctps_thr,
        primary_source=(src1000[primary][::step] * sign[primary]).astype(np.float32),
        source_srate=CARACAS_SRATE, source_t0=t0, bads=np.array(bads), analysis_window=np.array([t0, t1]))
    return stats


def removed_signal(mff_path):
    subj, sess = session_key(mff_path)
    key = f"{subj}/{sess}"
    stem = os.path.join(OUT_DIR, f"{subj}_{sess}")
    meta = np.load(stem + ".npz")
    ica = mne.preprocessing.read_ica(stem + "-ica.fif", verbose="error")
    t0, t1 = meta["analysis_window"]
    raw = load_eeg(mff_path, t0, t1, cfg.MANUAL_BAD_SPANS.get(key, []))
    raw.info["bads"] = [str(b) for b in meta["bads"]]
    raw.set_eeg_reference("average", verbose="error")
    raw.apply_function(lambda x: x - x.mean(), picks="eeg")
    hp = raw.copy().filter(APPLY_HP_HZ, None, verbose="error")
    clean = ica.apply(hp.copy(), exclude=[int(c) for c in meta["cardiac"]], verbose="error")
    return hp.get_data() - clean.get_data(), raw, ica


def all_sessions():
    for subj in sorted(os.listdir(cfg.RAW_DATA_DIR)):
        eeg = os.path.join(cfg.RAW_DATA_DIR, subj, "EEG")
        if not os.path.isdir(eeg):
            continue
        for sess in cfg.SESSIONS:
            if excluded(cfg.SUBJECT_ID.get(subj, subj), sess):
                continue
            mff = find_mff_file(eeg, sess)
            if mff:
                yield mff


def main(argv):
    force = "--force" in argv
    argv = [a for a in argv if a != "--force"]
    paths = list(all_sessions()) if argv == ["all"] else argv
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    rows = []
    for mff in paths:
        subj, sess = session_key(mff)
        if not force and os.path.exists(os.path.join(OUT_DIR, f"{subj}_{sess}.npz")) \
                and os.path.exists(summary_path):
            prev = pd.read_csv(summary_path)
            prev = prev[prev["session"] == f"{subj}/{sess}"]
            if len(prev) and prev["status"].iloc[0] == "ok":
                rows.append(prev.iloc[0].to_dict())
                print(f"{subj}/{sess}: done already, skipped")
                continue
        try:
            stats = run_session(mff)
            stats["status"] = "ok"
        except Exception as exc:
            stats = {"session": f"{subj}/{sess}", "status": f"FAILED {exc!r}"}
        stats["mff"] = mff
        rows.append(stats)
        print(f"{subj}/{sess}: " + ", ".join(f"{k}={v}" for k, v in stats.items() if k != "mff"), flush=True)
        os.makedirs(OUT_DIR, exist_ok=True)
        done = pd.DataFrame(rows)
        if os.path.exists(summary_path):
            prev = pd.read_csv(summary_path)
            done = pd.concat([prev[~prev["session"].isin(done["session"])], done], ignore_index=True)
        done.sort_values("session").to_csv(summary_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["all"])
