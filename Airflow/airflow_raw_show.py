#!/usr/bin/env python3
"""
Airflow raw signal viewer.

Two views per subject/session:
  1. Individual trial grid  — raw airflow epoch per trial, with optional NK2-
     cleaned overlay, breath peaks/troughs, and pass/reject annotations.
  2. Full session overview  — NK2-filtered session signal with startle markers.

Flags:
  --nk2        Overlay the NK2-cleaned (khodadad2018) signal on each trial (default ON).
  --raw-only   Show only the unfiltered raw signal (suppresses NK2 overlay).

Usage (from repo root):
  .venv/bin/python Airflow/airflow_raw_show.py --subject AB22
  .venv/bin/python Airflow/airflow_raw_show.py --subject AB22 --session eve --raw-only
  .venv/bin/python Airflow/airflow_raw_show.py --subject all --nk2

Output PNGs are saved to: <OUTPUT_DIR>/raw_show/<subject>/
"""

import argparse
import os
import pickle
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from Airflow import airflow_config as config
from Airflow import airflow_processor as processor

NCOLS = 6


# ── Cache loading with old-format backward-compat ────────────────────────────

def load_cache(cache_path, cfg):
    """
    Load airflow cache and return (trials_data, traces).

    Old cache format (key='trials'): trial dicts already have epoch_raw_wide and
    epoch_clean_wide, but rejection/score fields are None. We call _run_analysis
    directly on the existing trial dicts to populate them.

    New cache format (key='sessions'): pass through apply_analysis_params normally.
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    if "sessions" in cache:
        print("Cache format: new (sessions) — running apply_analysis_params")
        td, traces = processor.apply_analysis_params(cache["sessions"], cfg)
        return td, traces

    # Old format
    print("Cache format: old (trials) — running _run_analysis on cached epochs")
    raw_trials = cache["trials"]
    traces     = cache.get("traces", {})
    td         = {}

    for subj, sess_dict in sorted(raw_trials.items()):
        td[subj] = {}
        for sess_key, trial_list in sess_dict.items():
            # Shallow-copy so we don't mutate the cached dicts
            trials = [dict(t) for t in trial_list]
            # Reset analysis fields before re-running
            for t in trials:
                t["baseline_mean"]    = None
                t["score"]            = None
                t["rejected"]         = None
                t["rejection_reason"] = ""
                t["epoch_anal"]       = None
                t["times_anal"]       = None
            processor._run_analysis(trials, cfg)
            td[subj][sess_key] = trials

    return td, traces


# ── View 1: individual trial grid ─────────────────────────────────────────────

def plot_trials(trials, subj, sess_key, sess_label, outdir, show_nk2, cfg):
    n     = len(trials)
    nrows = max(1, (n + NCOLS - 1) // NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 3.0, nrows * 2.2),
                             squeeze=False)
    axes_flat = axes.flatten()

    label_name = {1: "Neg", 2: "Neu"}
    label_col  = {1: cfg.NEG_COLOR, 2: cfg.NEU_COLOR}

    for i, t in enumerate(trials):
        ax  = axes_flat[i]
        tw  = t["times_wide"].astype(float)
        ep_raw   = t["epoch_raw_wide"].astype(float)
        ep_clean = t.get("epoch_clean_wide")
        ep_peaks   = t.get("ep_peaks",   np.array([], dtype=int))
        ep_troughs = t.get("ep_troughs", np.array([], dtype=int))

        rej = t.get("rejected")
        lbl = t.get("label", 0)
        bm  = t.get("baseline_mean")
        sc  = t.get("score")
        rr  = t.get("rejection_reason", "")

        # Colour logic
        if rej:
            nk2_col   = "#95A5A6"
            title_col = "dimgray"
        else:
            nk2_col   = label_col.get(lbl, "#95A5A6")
            title_col = "black"

        # Raw signal (always grey background)
        ax.plot(tw, ep_raw, color="#CCCCCC", lw=0.7, zorder=2, label="raw")

        # NK2 overlay
        if show_nk2 and ep_clean is not None:
            ax.plot(tw, ep_clean.astype(float), color=nk2_col, lw=1.1, zorder=3, label="NK2")

            # Peaks and troughs on NK2 signal
            valid_peaks   = ep_peaks[(ep_peaks >= 0) & (ep_peaks < len(ep_clean))]
            valid_troughs = ep_troughs[(ep_troughs >= 0) & (ep_troughs < len(ep_clean))]
            if len(valid_peaks) > 0:
                ax.plot(tw[valid_peaks],   ep_clean[valid_peaks],
                        "^", color=nk2_col, ms=3.5, zorder=4, alpha=0.8)
            if len(valid_troughs) > 0:
                ax.plot(tw[valid_troughs], ep_clean[valid_troughs],
                        "v", color=nk2_col, ms=3.5, zorder=4, alpha=0.8)

        # Window shading
        ax.axvspan(cfg.BASELINE_TMIN,  cfg.BASELINE_TMAX,  alpha=0.10, color="#3498DB", zorder=0)
        ax.axvspan(cfg.RESPONSE_TMIN,  cfg.RESPONSE_TMAX,  alpha=0.08, color="#E74C3C",  zorder=0)

        # Baseline mean
        if bm is not None and not np.isnan(float(bm)):
            ax.axhline(float(bm), color="goldenrod", ls=":", lw=0.9, alpha=0.85)

        # Startle onset
        ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.45)

        ax.set_xlim(tw[0], tw[-1])
        ax.tick_params(labelsize=5.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Title
        if rej:
            title = f"T{i+1} REJ\n{rr[:22] or '?'}"
        else:
            sc_str = f"{sc:.2f}" if sc is not None and not np.isnan(float(sc)) else "—"
            title  = f"T{i+1} {label_name.get(lbl, '?')} sc={sc_str}"
        ax.set_title(title, fontsize=6.5, color=title_col, pad=2)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    nk2_note = " | blue=NK2 clean  grey=raw" if show_nk2 else " | raw only"
    fig.suptitle(
        f"{subj} — {sess_label} — Individual trials  (N={n}){nk2_note}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    path = os.path.join(outdir, f"{subj}_{sess_key}_trials.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── View 2: whole session overview ────────────────────────────────────────────

def plot_session(trace, startle_sec, trials, subj, sess_key, sess_label, outdir, cfg):
    sig    = trace["signal"].astype(float)
    sfreq  = trace["sfreq"]
    t_axis = np.arange(len(sig)) / sfreq

    fig, ax = plt.subplots(figsize=(20, 3.5))
    ax.plot(t_axis, sig, color="#2C3E50", lw=0.6, zorder=2, label="NK2 filtered")

    n = min(len(startle_sec), len(trials))
    for idx in range(n):
        t_star = float(startle_sec[idx])
        rej    = trials[idx].get("rejected")
        col    = "#27AE60" if not rej else "#E74C3C"
        ax.axvline(t_star, color=col, lw=0.9, alpha=0.75, zorder=4)
        ax.axvspan(t_star + cfg.WIDE_TMIN, t_star + cfg.WIDE_TMAX,
                   alpha=0.04, color="steelblue", zorder=0)

    ax.set_xlabel("Time in session (s)", fontsize=9)
    ax.set_ylabel("Airflow (a.u.)", fontsize=9)
    n_rej = sum(1 for t in trials[:n] if t.get("rejected"))
    ax.set_title(
        f"{subj} — {sess_label} — Full session (NK2)  "
        f"(green=pass, red=reject  {n_rej}/{n} rejected)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(outdir, f"{subj}_{sess_key}_session.png")
    plt.savefig(path, dpi=cfg.PLOT_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Airflow raw signal viewer")
    parser.add_argument("--subject",  required=True,
                        help="Subject ID (e.g. AB22) or 'all'")
    parser.add_argument("--session",  default=None, choices=["eve", "mor"],
                        help="Session to show (default: both)")
    parser.add_argument("--nk2",      action="store_true", default=True,
                        help="Show NK2-cleaned signal overlay (default ON)")
    parser.add_argument("--raw-only", action="store_true", default=False,
                        help="Show raw signal only, no NK2 overlay")
    args = parser.parse_args()

    show_nk2 = args.nk2 and not args.raw_only

    cache_path = os.path.join(config.OUTPUT_DIR, "_cache", "airflow_cache.pkl")
    if not os.path.exists(cache_path):
        print(f"Cache not found: {cache_path}\nRun airflow_main.py first.")
        sys.exit(1)

    print(f"Loading {cache_path} ...")
    td, traces = load_cache(cache_path, config)

    subjects = sorted(td.keys()) if args.subject == "all" else [args.subject]

    for subj in subjects:
        if subj not in td:
            print(f"Subject '{subj}' not in cache. Available: {sorted(td.keys())}")
            continue

        outdir = os.path.join(config.OUTPUT_DIR, "raw_show", subj)
        os.makedirs(outdir, exist_ok=True)

        sess_keys = ([args.session] if args.session
                     else sorted(k for k in td[subj] if td[subj][k]))

        for sess_key in sess_keys:
            trials = td[subj].get(sess_key, [])
            if not trials:
                print(f"  {subj} {sess_key}: no trials — skipping")
                continue

            sess_label = config.SESSION_MAP.get(sess_key, {}).get("label", sess_key)
            n_rej      = sum(1 for t in trials if t.get("rejected"))
            print(f"\n{subj} {sess_label}: {len(trials)} trials, {n_rej} rejected")

            # View 1 — individual trial grid
            plot_trials(trials, subj, sess_key, sess_label, outdir, show_nk2, config)

            # View 2 — full session
            tr = traces.get(subj, {}).get(sess_key)
            if tr is not None and tr.get("signal") is not None:
                startle_sec = tr["startle_samps_sec"]
                n = min(len(startle_sec), len(trials))
                plot_session(tr, startle_sec[:n], trials[:n],
                             subj, sess_key, sess_label, outdir, config)
            else:
                print(f"  No session trace for {subj} {sess_key} — skipping session view")

    print("\nDone. Open output folder:")
    print(f"  {os.path.join(config.OUTPUT_DIR, 'raw_show')}")


if __name__ == "__main__":
    main()
