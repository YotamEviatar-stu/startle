"""
Airflow Pipeline
================
Run with:
    .venv/bin/python Airflow/airflow_main.py

Layer 1 (MFF -> cache): ground-truth trial epochs (trial_epochs.build_trial_epochs)
+ raw signal downsampled to CACHE_SFREQ Hz.
Layer 2 (cache -> analysis): NK2 filter, baseline, rejection, scoring.

Only FORCE_RELOAD=True triggers a full MFF re-read. Everything else
(filter method, rejection thresholds) is re-applied every run. Epoch/baseline/
response window timing is ground truth (see trial_epochs.py), not a
config knob -- there's nothing left in Layer 2 to "re-apply" for window timing.

GLM scoring path (airflow_glm.py) uses two copies of the signal:
  - raw_z:      filtered copy. Only used to find WHEN each breath starts
                (a list of onset times). RP = time between onsets.
  - signal_raw: unfiltered cached signal. Once we know WHEN from raw_z, we
                measure HOW BIG each breath was (RA, then RFR = RA/RP) on
                THIS copy, so filtering doesn't shrink the real amplitude.
"""

import os
import sys
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))       # repo root
sys.path.insert(0, os.path.dirname(__file__))                           # Airflow/

import airflow_config as cfg
from airflow_glm import process_session
from airflow_amp_processor import apply_analysis_params
from extras.emg_raw_potentiation import find_startle_output_folder, find_mff_candidates, find_csv_by_suffix
from extras.trial_epochs import load_all_trials_ratings, TriggerAlignmentError

warnings.filterwarnings("ignore", category=RuntimeWarning)

CACHE_DIR  = os.path.join(cfg.OUTPUT_DIR, "_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "airflow_cache.pkl")


# ── Cache helpers ─────────────────────────────────────────────────────────────

def load_cache():
    if os.path.exists(CACHE_FILE) and not cfg.FORCE_RELOAD:
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "sessions" in data:
                return data
            print("  Cache is pre-migration format — rebuilding.")
        except Exception as e:
            print(f"  [!] Cache load failed ({e}) — rebuilding.")
    return {"sessions": {}}


def save_cache(data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


# ── Output helpers ────────────────────────────────────────────────────────────

def save_trial_csv(trials_data, output_dir):
    rows = []
    for subj, sessions in sorted(trials_data.items()):
        for sess_key, trials in sessions.items():
            for i, t in enumerate(trials, start=1):
                row = {
                    "subject":          subj,
                    "session":          sess_key,
                    "trial":            i,
                    "label":            t["label"],
                    "image_detail":     t["image_detail"],
                    "valence":          t["valence"],
                    "arousal":          t["arousal"],
                    "rejected":         t["rejected"],
                    "rejection_reason": t.get("rejection_reason", ""),
                }
                if "score_rp" in t:
                    row["score_rp"]  = t.get("score_rp")
                    row["score_ra"]  = t.get("score_ra")
                    row["score_rfr"] = t.get("score_rfr")
                else:
                    row["baseline_mean"] = t.get("baseline_mean")
                    row["score"]         = t.get("score")
                rows.append(row)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "airflow_trial_scores.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved trial CSV → {path}")


def plot_review_trials(trials_data, output_dir):
    """One PNG per trial: filtered epoch with BxB baseline and response cycles marked."""
    base_dir = os.path.join(output_dir, "review_trials")
    for subj, sessions in sorted(trials_data.items()):
        for sess_key, trials in sessions.items():
            if not trials:
                continue
            sess_dir = os.path.join(base_dir, subj, sess_key)
            os.makedirs(sess_dir, exist_ok=True)
            for i, t in enumerate(trials, start=1):
                if t["epoch_anal"] is None or t["times_anal"] is None:
                    continue

                tw = t["times_anal"]
                ep = t["epoch_anal"]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(tw, ep, color="black", linewidth=1.5, zorder=3)

                # Baseline quality-check window (gold) — used by rejection gates.
                # Ground truth is tw<0 (full pre-code span, variable length per
                # trial); tw[0] is that span's actual start as trimmed to the
                # display window, not a fixed config offset.
                ax.axvspan(tw[0], 0.0,
                           color="gold", alpha=0.12, label="Baseline window (rejection gate)")

                # BxB response breath: trough → peak (both after t=0)
                r_trough = t.get("response_trough_time")
                r_peak   = t.get("response_peak_time")
                if (r_trough is not None and r_peak is not None
                        and np.isfinite(r_trough) and np.isfinite(r_peak)):
                    ax.axvspan(r_trough, r_peak,
                               color="green", alpha=0.18, label="BxB response breath")
                    # Mark trough and peak on the signal
                    for t_time, marker, color in [(r_trough, "v", "green"), (r_peak, "^", "green")]:
                        idx = np.argmin(np.abs(tw - t_time))
                        if 0 <= idx < len(ep):
                            ax.plot(tw[idx], ep[idx], marker=marker,
                                    color=color, markersize=8, zorder=5)

                # BxB baseline peak (vertical dashed line)
                b_peak = t.get("baseline_peak_time")
                if b_peak is not None and np.isfinite(b_peak):
                    ax.axvline(b_peak, color="goldenrod", linestyle=":",
                               linewidth=1.5, label="BxB baseline peak")
                    idx = np.argmin(np.abs(tw - b_peak))
                    if 0 <= idx < len(ep):
                        ax.plot(tw[idx], ep[idx], "^", color="goldenrod",
                                markersize=8, zorder=5)

                ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Probe (t=0)")

                label_str = "Neg" if t["label"] == 1 else "Neu"
                score_str = (f"{t['score']:.3f}" if not t["rejected"]
                             and t["score"] is not None
                             and not np.isnan(t["score"]) else "--")
                b_amp_str = (f"{t['baseline_amp']:.3f}"
                             if t.get("baseline_amp") is not None
                             and np.isfinite(t["baseline_amp"]) else "--")
                r_amp_str = (f"{t['response_amp']:.3f}"
                             if t.get("response_amp") is not None
                             and np.isfinite(t["response_amp"]) else "--")
                ax.set_title(
                    f"{subj} {sess_key.upper()} Trial {i} | {label_str} | "
                    f"Score={score_str}  (b_amp={b_amp_str} → r_amp={r_amp_str}) | "
                    f"{'REJ: ' + t.get('rejection_reason', '') if t['rejected'] else 'OK'}"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Airflow (RSP_Clean)")
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, linestyle="--", alpha=0.3)

                tag   = "REJ" if t["rejected"] else "OK"
                fname = os.path.join(sess_dir, f"trial_{i:03d}_{tag}.png")
                fig.savefig(fname, dpi=cfg.PLOT_DPI, bbox_inches="tight")
                plt.close(fig)
    print(f"  Saved review trials → {base_dir}")


def plot_review_trials_glm(trials_data, output_dir):
    """One PNG per trial: z-scored airflow window (raw_z, left axis) with the
    RA feature series and its GLM fit on a twin right axis -- these are
    different quantities (raw breathing waveform vs. the derived per-cycle
    amplitude feature) with different natural scales, so they get separate
    axes rather than being overlaid on one. GLM-path analogue of
    plot_review_trials above."""
    base_dir = os.path.join(output_dir, "review_trials_glm")
    for subj, sessions in sorted(trials_data.items()):
        for sess_key, trials in sessions.items():
            if not trials:
                continue
            sess_dir = os.path.join(base_dir, subj, sess_key)
            os.makedirs(sess_dir, exist_ok=True)
            for i, t in enumerate(trials, start=1):
                if t["epoch_anal"] is None or t["times_anal"] is None:
                    continue

                tw = t["times_anal"]
                ep = t["epoch_anal"]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(tw, ep, color="black", linewidth=1.2, zorder=3, label="raw_z (Phase 1)")

                ax.axvspan(tw[0], 0.0, color="gold", alpha=0.12, label="Baseline window")
                ax.axvspan(0.0, tw[-1], color="green", alpha=0.08, label="Response window")
                ax.axvline(0, color="k", linestyle="--", linewidth=1, label="Picture onset (t=0)")
                if t.get("has_sound") and t.get("d110_rel") is not None:
                    ax.axvline(t["d110_rel"], color="red", linestyle="-", linewidth=1.5,
                               label="Sound (D110)")

                if t.get("series_ra") is not None and t.get("fitted_ra") is not None:
                    ax2 = ax.twinx()
                    ax2.plot(tw, t["series_ra"], color="steelblue", linewidth=1.2,
                             alpha=0.8, label="RA series (Phase 3)")
                    ax2.plot(tw, t["fitted_ra"], color="orange", linewidth=2,
                             label="GLM fit (RA)")
                    ax2.set_ylabel("RA feature (Phase-3 units)", color="steelblue")
                    ax2.legend(fontsize=7, loc="lower right")

                def fmt(v):
                    return f"{v:.3f}" if v is not None and np.isfinite(v) else "--"

                label_str = "Neg" if t["label"] == 1 else "Neu"
                ax.set_title(
                    f"{subj} {sess_key.upper()} Trial {i} | {label_str} | "
                    f"RP={fmt(t.get('score_rp'))} RA={fmt(t.get('score_ra'))} "
                    f"RFR={fmt(t.get('score_rfr'))} | "
                    f"{'REJ: ' + t.get('rejection_reason', '') if t['rejected'] else 'OK'}"
                )
                ax.set_xlabel("Time (s, re picture onset)")
                ax.set_ylabel("Airflow (z-scored)")
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, linestyle="--", alpha=0.3)

                tag   = "REJ" if t["rejected"] else "OK"
                fname = os.path.join(sess_dir, f"trial_{i:03d}_{tag}.png")
                fig.savefig(fname, dpi=cfg.PLOT_DPI, bbox_inches="tight")
                plt.close(fig)
    print(f"  Saved GLM review trials → {base_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if os.path.isdir(cfg.RAW_DATA_DIR):
        subjects = [d for d in sorted(os.listdir(cfg.RAW_DATA_DIR))
                    if os.path.isdir(os.path.join(cfg.RAW_DATA_DIR, d))]
        if cfg.SUBJECT_FILTER:
            subjects = [s for s in subjects if s in cfg.SUBJECT_FILTER]
    else:
        subjects = []

    cache            = load_cache()
    sessions_cache   = cache.setdefault("sessions", {})
    changed          = False

    # If drive is absent and cache already has data, skip MFF loading entirely.
    if not os.path.isdir(cfg.RAW_DATA_DIR):
        if sessions_cache:
            print(f"  Drive not mounted — using cache ({len(sessions_cache)} subjects).")
            subjects = []
        else:
            raise FileNotFoundError(
                f"RAW_DATA_DIR not found and cache is empty: {cfg.RAW_DATA_DIR}"
            )

    for subj in subjects:
        subj_path  = os.path.join(cfg.RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        try:
            startle_folder = find_startle_output_folder(subj_path)
        except OSError as e:
            print(f"[!] Cannot access {subj} ({e}) — skipping")
            continue
        if not startle_folder or not os.path.isdir(eeg_folder):
            print(f"[!] Missing data folders for {subj}")
            continue

        print(f"\n{subj}")
        sessions_cache.setdefault(subj, {})

        for sess_key, sess_meta in cfg.SESSION_MAP.items():
            if sess_key in sessions_cache[subj] and not cfg.FORCE_RELOAD:
                n = len(sessions_cache[subj][sess_key].get("trials_meta", []))
                print(f"  {sess_key}: using cache ({n} trials)")
                continue

            csv_path = find_csv_by_suffix(startle_folder, sess_meta["csv_suffix"])
            mff_candidates = find_mff_candidates(eeg_folder, sess_key)
            if not csv_path or not mff_candidates:
                print(f"  [!] Missing files for {subj} {sess_key}")
                continue

            ratings_df = load_all_trials_ratings(csv_path)
            # A subject can have more than one "*_{sess_key}_*.mff" on disk
            # (e.g. an aborted-then-restarted recording saved as "..._2") --
            # try each candidate in turn and use the first that actually
            # loads and aligns, instead of assuming the alphabetically-first
            # match is the right (or even readable) one.
            session = None
            for i, mff_path in enumerate(mff_candidates):
                try:
                    session = process_session(mff_path, ratings_df, config=cfg)
                except TriggerAlignmentError as e:
                    print(f"  [!] {subj} {sess_key}: trigger/CSV alignment failed for "
                          f"{os.path.basename(mff_path)} ({e})")
                    session = None
                if session is not None:
                    if i > 0:
                        print(f"  [i] {subj} {sess_key}: used fallback candidate "
                              f"{os.path.basename(mff_path)} (earlier match(es) failed)")
                    break
            if session is not None:
                sessions_cache[subj][sess_key] = session
                changed = True
            else:
                print(f"  [!] {subj} {sess_key}: no candidate .mff loaded — skipping this session")

    if changed:
        save_cache({"sessions": sessions_cache})
        print(f"\nSaved cache → {CACHE_FILE}")

    use_glm = getattr(cfg, "SCORING_METHOD", "peak_excursion_normalized") == "glm_deconvolution"

    if use_glm:
        from airflow_glm import run_glm_scoring
        print("\nRunning Layer 2 analysis (GLM/deconvolution scoring) ...")
        trials_data, _ = run_glm_scoring(sessions_cache, cfg)
    else:
        print("\nRunning Layer 2 analysis (NK2 filter + baseline + scoring) ...")
        trials_data, _ = apply_analysis_params(sessions_cache, cfg)

    # Generate plots only
    if use_glm:
        plot_review_trials_glm(trials_data, cfg.OUTPUT_DIR)
    else:
        plot_review_trials(trials_data, cfg.OUTPUT_DIR)

    print("\nTrial counts by condition:")
    for cond in ["Eve-Neg", "Eve-Neu", "Mor-Neg", "Mor-Neu"]:
        sk    = "eve" if cond.startswith("Eve") else "mor"
        label = 1 if cond.endswith("Neg") else 2
        all_t = [t for s in trials_data.values() for t in s.get(sk, [])
                 if t["label"] == label]
        acc   = sum(1 for t in all_t if not t["rejected"])
        print(f"  {cond}: {len(all_t)} total, {acc} accepted")

    print()
    for subj, sessions in sorted(trials_data.items()):
        for sess_key, trials in sessions.items():
            if not trials:
                continue
            n_rej = sum(1 for t in trials if t["rejected"])
            reasons = {}
            for t in trials:
                r = t.get("rejection_reason", "")
                if r:
                    reasons[r] = reasons.get(r, 0) + 1
            reason_str = "  " + ", ".join(f"{r}×{n}" for r, n in reasons.items()) if reasons else ""
            print(f"  {subj} {sess_key}: {len(trials)} trials, {n_rej} rejected{reason_str}")


if __name__ == "__main__":
    main()
