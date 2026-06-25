"""
Airflow Pipeline
================
Run from the repo root:
    .venv/bin/python -m Airflow.airflow_main

Layer 1 (MFF → cache): raw signal downsampled to CACHE_SFREQ Hz + trial metadata.
Layer 2 (cache → analysis): NK2 filter, epoch cutting, baseline, rejection, scoring.

Only FORCE_RELOAD=True triggers a full MFF re-read. Everything else
(filter method, epoch windows, rejection thresholds) is re-applied every run.
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Airflow.airflow_config as cfg
from Airflow.airflow_processor import (
    find_startle_output_folder,
    find_mff_file,
    find_csv_by_suffix,
    load_and_classify_ratings,
    process_session,
    apply_analysis_params,
)

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
                rows.append({
                    "subject":          subj,
                    "session":          sess_key,
                    "trial":            i,
                    "label":            t["label"],
                    "image_detail":     t["image_detail"],
                    "valence":          t["valence"],
                    "arousal":          t["arousal"],
                    "baseline_mean":    t["baseline_mean"],
                    "score":            t["score"],
                    "rejected":         t["rejected"],
                    "rejection_reason": t.get("rejection_reason", ""),
                })
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "airflow_trial_scores.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved trial CSV → {path}")


def plot_review_trials(trials_data, output_dir):
    """One PNG per trial: filtered epoch with baseline and response windows marked."""
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
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t["times_anal"], t["epoch_anal"], color="black", linewidth=1.5)
                ax.axvspan(cfg.BASELINE_TMIN, cfg.BASELINE_TMAX,
                           color="gold", alpha=0.12, label="Baseline")
                ax.axvspan(cfg.RESPONSE_TMIN, cfg.RESPONSE_TMAX,
                           color="green", alpha=0.10, label="Response window")
                ax.axvline(0, color="k", linestyle="--", linewidth=1)
                label_str = "Neg" if t["label"] == 1 else "Neu"
                score_str = (f"{t['score']:.3f}" if not t["rejected"]
                             and t["score"] is not None
                             and not np.isnan(t["score"]) else "--")
                ax.set_title(
                    f"{subj} {sess_key.upper()} Trial {i} | {label_str} | "
                    f"Score={score_str} | {'REJ: ' + t.get('rejection_reason','') if t['rejected'] else 'OK'}"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Airflow (RSP_Clean)")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.3)
                tag   = "REJ" if t["rejected"] else "OK"
                fname = os.path.join(sess_dir, f"trial_{i:03d}_{tag}.png")
                fig.savefig(fname, dpi=cfg.PLOT_DPI, bbox_inches="tight")
                plt.close(fig)
    print(f"  Saved review trials → {base_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    subjects = [d for d in sorted(os.listdir(cfg.RAW_DATA_DIR))
                if os.path.isdir(os.path.join(cfg.RAW_DATA_DIR, d))]
    if cfg.SUBJECT_FILTER:
        subjects = [s for s in subjects if s in cfg.SUBJECT_FILTER]

    cache            = load_cache()
    sessions_cache   = cache.setdefault("sessions", {})
    changed          = False

    for subj in subjects:
        subj_path      = os.path.join(cfg.RAW_DATA_DIR, subj)
        startle_folder = find_startle_output_folder(subj_path)
        eeg_folder     = os.path.join(subj_path, "EEG")
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
            mff_path = find_mff_file(eeg_folder, sess_key)
            if not csv_path or not mff_path:
                print(f"  [!] Missing files for {subj} {sess_key}")
                continue

            ratings_df = load_and_classify_ratings(csv_path)
            session    = process_session(mff_path, ratings_df, config=cfg)
            if session is not None:
                sessions_cache[subj][sess_key] = session
                changed = True

    if changed:
        save_cache({"sessions": sessions_cache})
        print(f"\nSaved cache → {CACHE_FILE}")

    print("\nRunning Layer 2 analysis (NK2 filter + baseline + scoring) ...")
    trials_data, _ = apply_analysis_params(sessions_cache, cfg)

    save_trial_csv(trials_data, cfg.OUTPUT_DIR)
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
