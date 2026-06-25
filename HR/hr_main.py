"""
HR Pipeline Main
===============
Orchestration: cache management, plotting, CSV export.

Run from repo root:
  .venv/bin/python -m HR.hr_main
  # or:
  .venv/bin/python HR/hr_main.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from HR import hr_config as config
from HR import hr_processor as processor

warnings.filterwarnings("ignore", category=RuntimeWarning)

CACHE_DIR  = os.path.join(config.OUTPUT_DIR, "_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "hr_cache.pkl")


# -- Cache helpers ------------------------------------------------------------

def load_cache():
    if os.path.exists(CACHE_FILE) and not config.FORCE_RELOAD:
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "sessions" in data:
                return data
            print("  Cache is pre-refactor format — rebuilding.")
        except Exception as e:
            print(f"  [!] Cache load failed ({e}) — rebuilding.")
    return {"sessions": {}}


def save_cache(data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)


# -- Helpers ------------------------------------------------------------------

def _sig_label(pval):
    return "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))


def _smooth(arr, sfreq, sigma_sec=4.0):
    """Gaussian smooth for display only."""
    return gaussian_filter1d(arr.astype(float), sigma=max(1, sigma_sec * sfreq))


def _pct_change(epoch, baseline):
    """Percent change from baseline. Returns absolute diff if baseline is 0."""
    if np.isnan(baseline) or baseline == 0:
        return epoch - baseline
    return (epoch - baseline) / baseline * 100


def _draw_stat_bracket(ax, x1, x2, y_top, pval, test_label, fontsize=10):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h, y_top + h, y_top], lw=1.2, color="black")
    ax.text((x1 + x2) / 2, y_top + h * 1.5,
            f"{_sig_label(pval)}  {test_label} p={pval:.3f}",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# -- Raw full-session trace (diagnostic) -------------------------------------

def plot_raw_sessions(subj_data_dict, traces_cache, output_dir):
    """
    Continuous SpO2-Pulse trace from session start to end, with a vertical
    line at each D110 startle event coloured by trial label.
    Uses the traces_cache built during MFF loading — no drive needed after
    the first run.
    """
    out_dir = os.path.join(output_dir, "raw_sessions")
    os.makedirs(out_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            if not trials:
                continue

            trace = traces_cache.get(subj, {}).get(sess_key)
            if trace is None:
                print(f"  [!] No cached trace for {subj} {sess_key} — skipping raw session plot")
                continue

            signal        = trace["signal"]
            sfreq         = trace["sfreq"]
            startle_times = trace["startle_samps_sec"]
            times         = np.arange(len(signal)) / sfreq
            smooth_sig    = _smooth(signal, sfreq, sigma_sec=6.0)

            y_lo = np.percentile(smooth_sig, 1) - 3
            y_hi = np.percentile(smooth_sig, 99) + 3

            fig, ax = plt.subplots(figsize=(22, 4))
            ax.plot(times, signal,     color="#AAAAAA", linewidth=0.5, alpha=0.5, zorder=1)
            ax.plot(times, smooth_sig, color="#222222", linewidth=1.2, alpha=0.9, zorder=2)

            for i, t_sec in enumerate(startle_times):
                if i < len(trials):
                    t_obj = trials[i]
                    clr = (config.REJ_COLOR if t_obj["rejected"]
                           else config.NEG_COLOR if t_obj["label"] == 1
                           else config.NEU_COLOR)
                else:
                    clr = "gray"
                ax.axvline(t_sec, color=clr, linewidth=1.2, alpha=0.75)

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel("Time in session (s)")
            ax.set_ylabel("HR (BPM)")
            sess_label = config.SESSION_MAP[sess_key]["label"]
            ax.set_title(f"{subj} — {sess_label} — Full HR Session", fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(handles=[
                plt.Line2D([0], [0], color=config.NEG_COLOR, label="Negative"),
                plt.Line2D([0], [0], color=config.NEU_COLOR, label="Neutral"),
                plt.Line2D([0], [0], color=config.REJ_COLOR, label="Rejected"),
            ], fontsize=9, loc="upper right")

            plt.tight_layout()
            fname = os.path.join(out_dir, f"{subj}_{sess_key}_raw_session.png")
            fig.savefig(fname, dpi=config.PLOT_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {fname}")


# -- Session timecourse -------------------------------------------------------

def plot_session_timecourses(subj_data_dict, output_dir):
    """One figure per subject: both sessions side-by-side, all trial epochs overlaid."""
    base_dir = os.path.join(output_dir, "session_timecourses")
    os.makedirs(base_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        fig.suptitle(f"{subj} — HR Session Timecourse", fontsize=13, fontweight="bold")
        has_any = False

        for ax, sess_key in zip(axes, ["eve", "mor"]):
            trials     = sessions.get(sess_key, [])
            sess_label = config.SESSION_MAP[sess_key]["label"]
            if not trials:
                ax.set_title(f"{sess_label} -- no data"); ax.axis("off"); continue
            has_any = True

            sfreq = trials[0]["epoch_wide"].shape[0] / (config.WIDE_TMAX - config.WIDE_TMIN)
            pct_epochs = [_pct_change(_smooth(t["epoch_wide"], sfreq), t["baseline_mean"])
                          for t in trials]
            y_lo, y_hi = -10, 10

            for t, pct_ep in zip(trials, pct_epochs):
                clr = config.REJ_COLOR if t["rejected"] else (
                      config.NEG_COLOR if t["label"] == 1 else config.NEU_COLOR)
                ax.plot(t["times_wide"], pct_ep, color=clr, linewidth=0.9, alpha=0.7)

            ax.set_ylim(y_lo, y_hi)
            ax.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.6)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold",  alpha=0.08)
            ax.axvspan(config.SCORE_TMIN,    config.SCORE_TMAX,    color="green", alpha=0.06)
            ax.set_title(sess_label, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time relative to startle (s)")
            ax.set_ylabel("HR % change from baseline")
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(handles=[
                plt.Line2D([0], [0], color=config.NEG_COLOR, label="Negative"),
                plt.Line2D([0], [0], color=config.NEU_COLOR, label="Neutral"),
                plt.Line2D([0], [0], color=config.REJ_COLOR, label="Rejected"),
            ], fontsize=8, loc="upper right")

        if has_any:
            plt.tight_layout()
            fname = os.path.join(base_dir, f"{subj}_session_timecourse.png")
            fig.savefig(fname, dpi=config.PLOT_DPI, bbox_inches="tight")
            print(f"  Saved: {fname}")
        plt.close(fig)


# -- Event-window timecourses -------------------------------------------------

def plot_event_windows(subj_data_dict, output_dir):
    """4-panel mean +/- SEM: Eve-Neg | Eve-Neu | Mor-Neg | Mor-Neu, baseline-corrected."""
    os.makedirs(output_dir, exist_ok=True)

    conditions = [
        ("eve", 1, "Evening -- Negative", config.NEG_COLOR),
        ("eve", 2, "Evening -- Neutral",  config.NEU_COLOR),
        ("mor", 1, "Morning -- Negative", config.NEG_COLOR),
        ("mor", 2, "Morning -- Neutral",  config.NEU_COLOR),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    fig.suptitle("HR Event Windows (baseline-corrected, mean +/- SEM)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()
    times_anal = None

    for ax, (sess_key, cond, title, color) in zip(axes, conditions):
        subj_avgs, n_trials_total = [], 0
        for subj, sessions in subj_data_dict.items():
            accepted = [t for t in sessions.get(sess_key, [])
                        if not t["rejected"] and t["label"] == cond]
            if not accepted:
                continue
            if times_anal is None:
                times_anal = accepted[0]["times_anal"]
            subj_avgs.append(np.mean([_pct_change(t["epoch_anal"], t["baseline_mean"]) for t in accepted], axis=0))
            n_trials_total += len(accepted)

        if not subj_avgs or times_anal is None:
            ax.set_title(f"{title} -- no data"); ax.axis("off"); continue

        arr = np.array(subj_avgs)
        avg = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0, nan_policy="omit") if len(subj_avgs) > 1 else np.zeros_like(avg)

        ax.plot(times_anal, avg, color=color, linewidth=2.2,
                label=f"n subj={len(subj_avgs)}, n trials={n_trials_total}")
        ax.fill_between(times_anal, avg - sem, avg + sem, color=color, alpha=0.20)
        ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
        ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold",  alpha=0.10)
        ax.axvspan(config.SCORE_TMIN,    config.SCORE_TMAX,    color="green", alpha=0.07)
        ax.set_ylim(-10, 10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("HR % change from baseline")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, "event_windows.png")
    fig.savefig(fname, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -- Per-trial review plots ---------------------------------------------------

def plot_review_trials(subj_data_dict, output_dir):
    base_dir = os.path.join(output_dir, "review_trials")
    os.makedirs(base_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        for sess_key, trials in sessions.items():
            if not trials:
                continue
            sess_dir = os.path.join(base_dir, subj, sess_key)
            os.makedirs(sess_dir, exist_ok=True)

            for i, t in enumerate(trials, start=1):
                rej  = t["rejected"]
                clr  = config.REJ_COLOR if rej else (config.NEG_COLOR if t["label"] == 1 else config.NEU_COLOR)
                cond = "REJECTED" if rej else ("Negative" if t["label"] == 1 else "Neutral")
                score_str = (f"{t['score']:.1f}%" if not rej and t["score"] is not None
                             and not np.isnan(t["score"]) else "--")

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t["times_wide"], t["epoch_wide"], color=clr, linewidth=1.5)
                ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold",  alpha=0.12, label="Baseline")
                ax.axvspan(config.SCORE_TMIN,    config.SCORE_TMAX,    color="green", alpha=0.10, label="Score window")
                ax.axvline(0, color="k", linestyle="--", linewidth=1)
                if not rej and t["baseline_mean"] is not None:
                    ax.axhline(t["baseline_mean"], color="gray", linestyle=":", linewidth=1,
                               label=f"Baseline = {t['baseline_mean']:.1f} bpm")
                ax.set_title(
                    f"{subj} {sess_key.upper()} Trial {i} | {cond} | "
                    f"Val={t['valence']} Ar={t['arousal']} | Score={score_str}"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("HR (BPM)")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle="--", alpha=0.3)

                tag   = "REJ" if rej else cond[:3].upper()
                fname = os.path.join(sess_dir, f"trial_{i:03d}_{tag}.png")
                fig.savefig(fname, dpi=100, bbox_inches="tight")
                plt.close(fig)

    print(f"  Saved review plots under {os.path.join(output_dir, 'review_trials')}")


# -- Group score boxplot (4 conditions) ---------------------------------------

def plot_group_boxplot(subj_data_dict, output_dir):
    """Boxplot + scatter: Eve-Neg | Eve-Neu | Mor-Neg | Mor-Neu."""
    os.makedirs(output_dir, exist_ok=True)

    cond_keys = [
        ("eve", 1, "Eve-Neg", config.NEG_COLOR),
        ("eve", 2, "Eve-Neu", config.NEU_COLOR),
        ("mor", 1, "Mor-Neg", "#922B21"),
        ("mor", 2, "Mor-Neu", "#1A5276"),
    ]

    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sk, cond, *_ in cond_keys:
            vals = [t["score"] for t in sessions.get(sk, [])
                    if not t["rejected"] and t["label"] == cond and not np.isnan(t["score"])]
            if vals:
                subj_means[subj][(sk, cond)] = np.mean(vals)

    valid = [s for s in subj_means if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    if len(valid) < 2:
        print("  [!] Not enough subjects for group boxplot")
        return

    data_cols = [[subj_means[s][(sk, c)] for s in valid] for sk, c, *_ in cond_keys]
    n_trials  = []
    for sk, cond, *_ in cond_keys:
        n_trials.append(sum(1 for s in valid
                            for t in subj_data_dict[s].get(sk, [])
                            if not t["rejected"] and t["label"] == cond))

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 3, 4]
    bp = ax.boxplot(data_cols, positions=positions, widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, (_, _, _, clr) in zip(bp["boxes"], cond_keys):
        patch.set_facecolor(clr); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)

    for s in valid:
        j = np.random.uniform(-0.07, 0.07)
        for (pos_n, pos_u), (sk_n, sk_u) in [((1, 2), ("eve", "eve")), ((3, 4), ("mor", "mor"))]:
            vn = subj_means[s].get((sk_n, 1)); vu = subj_means[s].get((sk_u, 2))
            if vn is not None and vu is not None:
                lc = config.UP_COLOR if vu >= vn else config.DOWN_COLOR
                ax.plot([pos_n+j, pos_u+j], [vn, vu], color=lc, alpha=0.45, linewidth=1.2)
                ax.scatter(pos_n+j, vn, color=cond_keys[0 if pos_n==1 else 2][3],
                           edgecolors="k", s=45, zorder=4, linewidths=0.5)
                ax.scatter(pos_u+j, vu, color=cond_keys[1 if pos_u==2 else 3][3],
                           edgecolors="k", s=45, zorder=4, linewidths=0.5)

    for pair_idx, (col_a, col_b, x1, x2) in enumerate(
        [(data_cols[0], data_cols[1], 1, 2), (data_cols[2], data_cols[3], 3, 4)]
    ):
        try:
            _, p = stats.wilcoxon(col_a, col_b); tl = "Wilcoxon"
        except Exception:
            _, p = stats.ttest_rel(col_a, col_b); tl = "Paired t"
        y_top = max(max(col_a), max(col_b)) * 1.08
        ax.set_ylim(top=max(ax.get_ylim()[1], y_top * 1.25))
        _draw_stat_bracket(ax, x1, x2, y_top, p, tl)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lbl}\n(n={nt})" for (_, _, lbl, _), nt in zip(cond_keys, n_trials)])
    ax.set_ylabel("HR % change from baseline")
    ax.set_title(f"HR Score by Condition (N={len(valid)} subjects)", fontweight="bold")
    ax.axvline(2.5, color="lightgray", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_boxplot.png")
    fig.savefig(fname, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


# -- CSV export ---------------------------------------------------------------

def save_trial_csv(subj_data_dict, output_dir):
    rows = []
    for subj, sessions in sorted(subj_data_dict.items()):
        for sess_key, trials in sessions.items():
            for i, t in enumerate(trials, start=1):
                rows.append({
                    "subject":        subj,
                    "session":        sess_key,
                    "trial":          i,
                    "label":          t["label"],
                    "image_detail":   t["image_detail"],
                    "valence":        t["valence"],
                    "arousal":        t["arousal"],
                    "baseline_mean":  t["baseline_mean"],
                    "score":          t["score"],
                    "rejected":       t["rejected"],
                })
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hr_trial_data.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Saved trial CSV: {csv_path}")


# -- Trial score scatter (per subject) ----------------------------------------

def plot_trial_scores_per_subject(subj_data_dict, output_dir):
    plot_dir = os.path.join(output_dir, "trial_score_scatter")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(f"{subj} — HR Trial Score by Trial Number", fontsize=13, fontweight="bold")

        for ax, sess_key in zip(axes, ["mor", "eve"]):
            trials = sessions.get(sess_key)
            if not trials:
                ax.set_title(f"{sess_key.upper()} — no data"); continue

            scores    = np.array([t["score"] for t in trials])
            labels    = np.array([t["label"] for t in trials])
            rejected  = np.array([t["rejected"] for t in trials])
            trial_nums = np.arange(1, len(trials) + 1)

            for cond, color, name in [(1, config.NEG_COLOR, "Negative"),
                                       (2, config.NEU_COLOR, "Neutral")]:
                mask = (~rejected) & (labels == cond)
                if mask.sum() > 0:
                    ax.scatter(trial_nums[mask], scores[mask], color=color,
                               s=60, edgecolors="white", linewidths=0.5, zorder=3,
                               label=f"{name} (n={mask.sum()})")
            n_rej = rejected.sum()
            if n_rej > 0:
                ax.scatter(trial_nums[rejected], np.zeros(n_rej),
                           color=config.REJ_COLOR, marker="x", s=80, zorder=2,
                           label=f"Rejected (n={n_rej})")

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(config.SESSION_MAP[sess_key]["label"], fontsize=11, fontweight="bold")
            ax.set_xlabel("Trial number")
            if ax is axes[0]:
                ax.set_ylabel("HR % change from baseline")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{subj}_trial_scores.png")
        fig.savefig(fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved trial-score scatter: {fname}")


# -- Per-subject average timecourse -------------------------------------------

def plot_subject_average_timecourse(subj_data_dict, output_dir):
    plot_dir = os.path.join(output_dir, "subject_average_timecourses")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(f"{subj} — Average HR Timecourse", fontsize=13, fontweight="bold")

        for ax, sess_key in zip(axes, ["eve", "mor"]):
            sess_label = config.SESSION_MAP[sess_key]["label"]
            trials = sessions.get(sess_key, [])

            neg_epochs, neu_epochs = [], []
            for t in trials:
                if t["rejected"] or t["epoch_anal"] is None:
                    continue
                delta = _pct_change(t["epoch_anal"], t["baseline_mean"])
                if t["label"] == 1:
                    neg_epochs.append(delta)
                else:
                    neu_epochs.append(delta)

            times = next((t["times_anal"] for t in trials if t["times_anal"] is not None), None)
            if times is None or (not neg_epochs and not neu_epochs):
                ax.set_title(f"{sess_label} — no data"); continue

            for epochs, color, name in [
                (neg_epochs, config.NEG_COLOR, "Negative"),
                (neu_epochs, config.NEU_COLOR, "Neutral"),
            ]:
                if not epochs:
                    continue
                arr = np.array(epochs)
                avg = np.mean(arr, axis=0)
                sem = stats.sem(arr, axis=0, nan_policy="omit") if len(arr) > 1 else np.zeros_like(avg)
                ax.plot(times, avg, color=color, linewidth=2.2,
                        label=f"{name} (n={len(epochs)})")
                ax.fill_between(times, avg - sem, avg + sem, color=color, alpha=0.20)

            ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
            ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
            ax.axvspan(config.SCORE_TMIN, config.SCORE_TMAX, color="green", alpha=0.07)
            ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold", alpha=0.07)
            ax.set_ylim(-10, 10)
            ax.set_title(sess_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (s)")
            if ax is axes[0]:
                ax.set_ylabel("HR % change from baseline")
                ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{subj}_average_timecourse.png")
        fig.savefig(fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved subject average timecourse: {fname}")


# -- Group average timecourse -------------------------------------------------

def plot_group_average_timecourse(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Group Average HR Timecourse (accepted trials)",
                 fontsize=13, fontweight="bold")

    for ax, sess_key in zip(axes, ["eve", "mor"]):
        sess_label = config.SESSION_MAP[sess_key]["label"]
        neg_epochs, neu_epochs = [], []
        n_subjs_neg = n_subjs_neu = 0
        times = None

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            subj_neg = [_pct_change(t["epoch_anal"], t["baseline_mean"])
                        for t in trials if not t["rejected"] and t["label"] == 1
                        and t["epoch_anal"] is not None]
            subj_neu = [_pct_change(t["epoch_anal"], t["baseline_mean"])
                        for t in trials if not t["rejected"] and t["label"] == 2
                        and t["epoch_anal"] is not None]
            if subj_neg:
                neg_epochs.append(np.mean(subj_neg, axis=0))
                n_subjs_neg += 1
            if subj_neu:
                neu_epochs.append(np.mean(subj_neu, axis=0))
                n_subjs_neu += 1
            if times is None:
                times = next((t["times_anal"] for t in trials if t["times_anal"] is not None), None)

        if times is None or (not neg_epochs and not neu_epochs):
            ax.set_title(f"{sess_label} — no data"); continue

        n_trials_neg = sum(1 for _, s in subj_data_dict.items()
                           for t in s.get(sess_key, []) if not t["rejected"] and t["label"] == 1)
        n_trials_neu = sum(1 for _, s in subj_data_dict.items()
                           for t in s.get(sess_key, []) if not t["rejected"] and t["label"] == 2)

        for epochs, color, name, n_subjs, n_trials in [
            (neg_epochs, config.NEG_COLOR, "Negative", n_subjs_neg, n_trials_neg),
            (neu_epochs, config.NEU_COLOR, "Neutral",  n_subjs_neu, n_trials_neu),
        ]:
            if not epochs:
                continue
            arr = np.array(epochs)
            avg = np.mean(arr, axis=0)
            sem = stats.sem(arr, axis=0, nan_policy="omit") if len(arr) > 1 else np.zeros_like(avg)
            ax.plot(times, avg, color=color, linewidth=2.2,
                    label=f"{name} (n subj={n_subjs}, n trials={n_trials})")
            ax.fill_between(times, avg - sem, avg + sem, color=color, alpha=0.20)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
        ax.axvspan(config.SCORE_TMIN, config.SCORE_TMAX, color="green", alpha=0.07,
                   label=f"Score window ({config.SCORE_TMIN}–{config.SCORE_TMAX} s)")
        ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold", alpha=0.07)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylim(-10, 10)
        ax.set_title(sess_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (s)")
        if ax is axes[0]:
            ax.set_ylabel("HR % change from baseline")
            ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(output_dir, "group_average_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group average timecourse: {fname}")


# -- Group Neg/Neu ratio boxplot ----------------------------------------------

def plot_group_ratio_boxplot(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    eve_ratios, mor_ratios = [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        ratios = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            neg = [t["score"] for t in trials if not t["rejected"] and t["label"] == 1 and not np.isnan(t["score"])]
            neu = [t["score"] for t in trials if not t["rejected"] and t["label"] == 2 and not np.isnan(t["score"])]
            if neg and neu and np.mean(neu) != 0:
                ratios[sess_key] = np.mean(neg) / np.mean(neu)
        if "eve" in ratios and "mor" in ratios:
            eve_ratios.append(ratios["eve"])
            mor_ratios.append(ratios["mor"])

    n = len(eve_ratios)
    if n < 2:
        print("  [!] Not enough subjects for ratio plot"); return

    try:
        _, pval = stats.wilcoxon(eve_ratios, mor_ratios); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_ratios, mor_ratios); tl = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_ratios, mor_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for ev, mo in zip(eve_ratios, mor_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(eve_ratios), max(mor_ratios)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n})", f"Morning\n(n={n})"], fontsize=11)
    ax.set_ylabel("Neg / Neu HR % change ratio", fontsize=11)
    ax.set_title(f"Negative-to-Neutral Ratio: Evening vs Morning\n(N={n} subjects)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_neg_neu_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Neg/Neu ratio boxplot: {fname}")


# -- Group session comparison (Eve vs Mor by condition) -----------------------

def plot_group_session_comparison(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cond_keys = [
        ("eve", 1, "Eve-Neg", config.NEG_COLOR),
        ("mor", 1, "Mor-Neg", "#922B21"),
        ("eve", 2, "Eve-Neu", config.NEU_COLOR),
        ("mor", 2, "Mor-Neu", "#1A5276"),
    ]
    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sk, cond, *_ in cond_keys:
            vals = [t["score"] for t in sessions.get(sk, [])
                    if not t["rejected"] and t["label"] == cond and not np.isnan(t["score"])]
            if vals:
                subj_means[subj][(sk, cond)] = np.mean(vals)

    valid = [s for s in subj_means if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for session comparison"); return

    data_cols = [[subj_means[s][(sk, c)] for s in valid] for sk, c, *_ in cond_keys]
    n_trials  = [sum(1 for subj in valid for t in subj_data_dict[subj].get(sk, [])
                     if not t["rejected"] and t["label"] == c)
                 for sk, c, *_ in cond_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions  = [1, 2, 3.6, 4.6]
    bp = ax.boxplot(data_cols, positions=positions, widths=0.42,
                    patch_artist=True, showfliers=False)
    for patch, (_, _, _, clr) in zip(bp["boxes"], cond_keys):
        patch.set_facecolor(clr); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)

    for s in valid:
        j = np.random.uniform(-0.07, 0.07)
        for (p1, p2), (k1, k2) in [((1, 2), (("eve", 1), ("mor", 1))),
                                     ((3.6, 4.6), (("eve", 2), ("mor", 2)))]:
            v1 = subj_means[s][k1]; v2 = subj_means[s][k2]
            lc = config.UP_COLOR if v2 >= v1 else config.DOWN_COLOR
            ax.plot([p1+j, p2+j], [v1, v2], color=lc, alpha=0.55, linewidth=1.4)
            ax.scatter(p1+j, v1, color=cond_keys[0 if p1==1 else 2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(p2+j, v2, color=cond_keys[1 if p2==2 else 3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    neg_eve, neg_mor, neu_eve, neu_mor = data_cols
    for col_a, col_b, x1, x2 in [(neg_eve, neg_mor, 1, 2), (neu_eve, neu_mor, 3.6, 4.6)]:
        try:
            _, p = stats.wilcoxon(col_a, col_b); tl = "Wilcoxon"
        except Exception:
            _, p = stats.ttest_rel(col_a, col_b); tl = "Paired t"
        y_top = max(max(col_a), max(col_b)) * 1.10
        ax.set_ylim(top=max(ax.get_ylim()[1], y_top * 1.20))
        _draw_stat_bracket(ax, x1, x2, y_top, p, tl)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{lbl}\n(n={nt})" for (_, _, lbl, _), nt in zip(cond_keys, n_trials)], fontsize=10)
    ax.set_ylabel("HR % change from baseline", fontsize=11)
    ax.set_title(f"Evening vs Morning by Condition  (N={n} subjects)\nGreen = score increased, Red = decreased",
                 fontsize=12, fontweight="bold")
    ax.axvline(2.8, color="lightgray", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_session_comparison_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved session comparison boxplot: {fname}")


# -- Group Eve/Mor ratio per condition ----------------------------------------

def plot_group_eve_mor_ratio(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    neg_ratios, neu_ratios = [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        r = {}
        for sess_key in ["eve", "mor"]:
            for cond in [1, 2]:
                vals = [t["score"] for t in sessions.get(sess_key, [])
                        if not t["rejected"] and t["label"] == cond and not np.isnan(t["score"])]
                if vals:
                    r[(sess_key, cond)] = np.mean(vals)
        if all(k in r for k in [("eve",1),("mor",1),("eve",2),("mor",2)]):
            if r[("mor",1)] != 0 and r[("mor",2)] != 0:
                neg_ratios.append(r[("eve",1)] / r[("mor",1)])
                neu_ratios.append(r[("eve",2)] / r[("mor",2)])

    n = len(neg_ratios)
    if n < 2:
        print("  [!] Not enough subjects for Eve/Mor ratio plot"); return

    try:
        _, pval = stats.wilcoxon(neg_ratios, neu_ratios); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(neg_ratios, neu_ratios); tl = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([neg_ratios, neu_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.NEG_COLOR, config.NEU_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for neg_r, neu_r in zip(neg_ratios, neu_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if neu_r >= neg_r else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [neg_r, neu_r], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, neg_r, color=config.NEG_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, neu_r, color=config.NEU_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(neg_ratios), max(neu_ratios)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Negative\n(n={n})", f"Neutral\n(n={n})"], fontsize=11)
    ax.set_ylabel("Evening / Morning HR % change ratio", fontsize=11)
    ax.set_title(f"Evening-to-Morning Ratio: Neg vs Neu  (N={n} subjects)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_eve_mor_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Eve/Mor ratio boxplot: {fname}")


# -- Overall Eve vs Mor -------------------------------------------------------

def plot_group_overall_eve_vs_mor(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    eve_means, mor_means, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sk in ["eve", "mor"]:
            sc = [t["score"] for t in sessions.get(sk, [])
                  if not t["rejected"] and not np.isnan(t["score"])]
            if sc:
                vals[sk] = np.mean(sc)
        if "eve" in vals and "mor" in vals:
            eve_means.append(vals["eve"]); mor_means.append(vals["mor"]); valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for overall Eve vs Mor"); return

    try:
        _, pval = stats.wilcoxon(eve_means, mor_means); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_means, mor_means); tl = "Paired t"

    nt_eve = sum(1 for s in valid for t in subj_data_dict[s].get("eve", []) if not t["rejected"])
    nt_mor = sum(1 for s in valid for t in subj_data_dict[s].get("mor", []) if not t["rejected"])

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_means, mor_means], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for ev, mo in zip(eve_means, mor_means):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(eve_means), max(mor_means)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n subj={n}, n trials={nt_eve})",
                        f"Morning\n(n subj={n}, n trials={nt_mor})"], fontsize=10)
    ax.set_ylabel("HR % change from baseline", fontsize=11)
    ax.set_title(f"Overall HR Score: Evening vs Morning\n(N={n} subjects, green=↑ red=↓)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_eve_vs_mor.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall Eve vs Mor: {fname}")


# -- Overall Eve/Mor ratio (vs 1) ---------------------------------------------

def plot_group_overall_ratio(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ratios, valid = [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sk in ["eve", "mor"]:
            sc = [t["score"] for t in sessions.get(sk, [])
                  if not t["rejected"] and not np.isnan(t["score"])]
            if sc:
                vals[sk] = np.mean(sc)
        if "eve" in vals and "mor" in vals and vals["mor"] != 0:
            ratios.append(vals["eve"] / vals["mor"]); valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for overall ratio"); return

    try:
        _, pval = stats.wilcoxon([r - 1 for r in ratios]); tl = "Wilcoxon (vs 1)"
    except Exception:
        _, pval = stats.ttest_1samp(ratios, 1); tl = "One-sample t (vs 1)"

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(ratios, positions=[1], widths=0.35, patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#95A5A6"); bp["boxes"][0].set_alpha(0.55); bp["boxes"][0].set_edgecolor("gray")
    bp["medians"][0].set(color="black", linewidth=2)
    for r in ratios:
        j = np.random.uniform(-0.08, 0.08)
        ax.scatter(1+j, r, color="#7F8C8D", edgecolors="k", s=60, zorder=3, linewidths=0.5)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.5)
    y_top = max(ratios) * 1.08
    ax.set_ylim(top=y_top * 1.20)
    sig = _sig_label(pval)
    ax.text(1, y_top * 1.05, f"{sig}\n{tl}\np={pval:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks([1])
    ax.set_xticklabels([f"Eve / Mor\n(n={n})"], fontsize=11)
    ax.set_ylabel("Evening / Morning HR % change ratio", fontsize=11)
    ax.set_title("Overall Eve/Mor Ratio (all accepted trials)\nRatio > 1 = Eve > Mor",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_ratio.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall ratio: {fname}")


# -- Overall average timecourse (Eve vs Mor) ----------------------------------

def plot_group_overall_timecourse(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    times = None

    for sess_key, sess_label, color in [("eve", "Evening", config.EVE_COLOR),
                                         ("mor", "Morning", config.MOR_COLOR)]:
        subj_avgs = []
        n_trials_total = 0
        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            accepted = [t for t in trials if not t["rejected"] and t["epoch_anal"] is not None]
            if not accepted:
                continue
            epochs = [_pct_change(t["epoch_anal"], t["baseline_mean"]) for t in accepted]
            subj_avgs.append(np.mean(epochs, axis=0))
            n_trials_total += len(accepted)
            if times is None:
                times = accepted[0]["times_anal"]

        if not subj_avgs or times is None:
            continue
        arr = np.array(subj_avgs)
        avg = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0, nan_policy="omit") if len(arr) > 1 else np.zeros_like(avg)
        ax.plot(times, avg, color=color, linewidth=2.5,
                label=f"{sess_label} (n subj={len(subj_avgs)}, n trials={n_trials_total})")
        ax.fill_between(times, avg - sem, avg + sem, color=color, alpha=0.20)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvspan(config.SCORE_TMIN, config.SCORE_TMAX, color="green", alpha=0.07,
               label=f"Score window ({config.SCORE_TMIN}–{config.SCORE_TMAX} s)")
    ax.axvspan(config.BASELINE_TMIN, config.BASELINE_TMAX, color="gold", alpha=0.07)
    ax.set_ylim(-10, 10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("HR % change from baseline", fontsize=11)
    ax.set_title("Overall Average HR Timecourse: Evening vs Morning\n(all accepted trials, ±SEM shaded)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall timecourse: {fname}")


# -- STAI-T correlations ------------------------------------------------------

def plot_stai_correlations(subj_data_dict, output_dir, xlsx_path=None):
    if xlsx_path is None:
        xlsx_path = getattr(config, "SUBJECTS_XLSX", None)
    if not xlsx_path:
        print("  [!] SUBJECTS_XLSX not set in config — skipping STAI correlations"); return
    try:
        stai_df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [!] Cannot read {xlsx_path}: {e}"); return

    stai_df["ID"] = stai_df["ID"].astype(str).str.strip().str.upper()
    stai_map = dict(zip(stai_df["ID"], stai_df["STAI-T"]))

    rows = []
    for subj, sessions in sorted(subj_data_dict.items()):
        stai = stai_map.get(subj.upper())
        if stai is None or pd.isna(stai):
            continue
        m = {"subj": subj, "stai": float(stai)}
        for sk, cl in [("eve","neg"),("eve","neu"),("mor","neg"),("mor","neu")]:
            cv = 1 if cl == "neg" else 2
            vals = [t["score"] for t in sessions.get(sk, [])
                    if not t["rejected"] and t["label"] == cv and not np.isnan(t["score"])]
            m[f"hr_{sk}_{cl}"] = np.mean(vals) if vals else np.nan
        for sk, lbl in [("eve","eve"),("mor","mor")]:
            neg_t = [t for t in sessions.get(sk, []) if not t["rejected"] and t["label"] == 1]
            m[f"val_{lbl}"]  = np.mean([t["valence"] for t in neg_t]) if neg_t else np.nan
            m[f"arsl_{lbl}"] = np.mean([t["arousal"] for t in neg_t]) if neg_t else np.nan
        rows.append(m)

    if len(rows) < 3:
        print("  [!] Not enough subjects with STAI-T for correlation plot"); return

    df = pd.DataFrame(rows).set_index("subj")
    panels = [
        ("hr_eve_neg", "Score (HR)", "Evening – Negative"),
        ("hr_eve_neu", "Score (HR)", "Evening – Neutral"),
        ("hr_mor_neg", "Score (HR)", "Morning – Negative"),
        ("hr_mor_neu", "Score (HR)", "Morning – Neutral"),
        ("val_eve",      "Valence",     "Valence – Evening (Neg trials)"),
        ("val_mor",      "Valence",     "Valence – Morning (Neg trials)"),
        ("arsl_eve",     "Arousal",     "Arousal – Evening (Neg trials)"),
        ("arsl_mor",     "Arousal",     "Arousal – Morning (Neg trials)"),
    ]
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axes = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(panels):
        ax = axes[idx]
        y = df[col].values.astype(float); x = df["stai"].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y, n = x[mask], y[mask], mask.sum()
        if n < 3 or np.std(x) == 0 or np.std(y) == 0:
            ax.set_title(title + "\n(insufficient data)", fontsize=10); ax.axis("off"); continue
        r, p = stats.pearsonr(x, y)
        dot_color = config.NEG_COLOR if "neg" in col.lower() else (config.NEU_COLOR if "neu" in col.lower() else "#7F8C8D")
        ax.scatter(x, y, color=dot_color, edgecolors="white", s=60, linewidths=0.5, zorder=3, alpha=0.85)
        m_c, b_c = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m_c * x_line + b_c, color="#2C3E50", linewidth=1.6, linestyle="--")
        for sid, xi, yi in zip(df.index[mask], x, y):
            ax.annotate(sid, (xi, yi), fontsize=6, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points", color="#555555")
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(0.04, 0.97, f"r={r:+.2f}  {p_str}  {_sig_label(p)}\nn={n}",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.85))
        ax.set_xlabel("STAI-T score", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("STAI-T Correlations — HR Startle & Subjective Ratings",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "stai_correlations.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved STAI-T correlations: {fname}")


# -- Subjective negative % (CSV-only) -----------------------------------------

def plot_subjective_negative_percentage(raw_data_dir, output_dir):
    subjects = sorted([d for d in os.listdir(raw_data_dir)
                       if os.path.isdir(os.path.join(raw_data_dir, d))])
    data = []
    for subj in subjects:
        if config.SUBJECT_FILTER and subj not in config.SUBJECT_FILTER:
            continue
        startle_folder = processor.find_startle_output_folder(os.path.join(raw_data_dir, subj))
        if not startle_folder:
            continue
        pcts = {}
        for sk, sess_cfg in config.SESSION_MAP.items():
            csv_path = processor.find_csv_by_suffix(startle_folder, sess_cfg["csv_suffix"])
            if not csv_path:
                continue
            try:
                df = processor.load_and_classify_ratings(csv_path)
                if len(df):
                    pcts[sk] = 100.0 * sum(df["subjective_label"] == 1) / len(df)
            except Exception:
                pass
        if "eve" in pcts and "mor" in pcts:
            data.append({"subj": subj, "eve_pct": pcts["eve"], "mor_pct": pcts["mor"]})

    n = len(data)
    if n < 2:
        print("  [!] Not enough subjects for negative% plot"); return

    df_pcts = pd.DataFrame(data)
    eve_vals = df_pcts["eve_pct"].values
    mor_vals = df_pcts["mor_pct"].values
    try:
        _, pval = stats.wilcoxon(eve_vals, mor_vals); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_vals, mor_vals); tl = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_vals, mor_vals], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for _, row in df_pcts.iterrows():
        ev, mo = row["eve_pct"], row["mor_pct"]
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(eve_vals), max(mor_vals)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n})", f"Morning\n(n={n})"], fontsize=11)
    ax.set_ylabel("% Subjective Negative trials", fontsize=11)
    ax.set_title("Subjective Negative Trials %: Evening vs Morning\n(based on CSV ratings only)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "subjective_negative_percentage_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subjective negative% boxplot: {fname}")


# -- Eve vs Mor boxplot (2 boxes, collapsed across conditions) ----------------

def plot_eve_vs_mor_boxplot(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    eve_means, mor_means, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sk in ["eve", "mor"]:
            sc = [t["score"] for t in sessions.get(sk, [])
                  if not t["rejected"] and not np.isnan(t["score"])]
            if sc:
                vals[sk] = np.mean(sc)
        if "eve" in vals and "mor" in vals:
            eve_means.append(vals["eve"]); mor_means.append(vals["mor"]); valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for Eve vs Mor boxplot"); return

    try:
        _, pval = stats.wilcoxon(eve_means, mor_means); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_means, mor_means); tl = "Paired t"

    nt_eve = sum(1 for s in valid for t in subj_data_dict[s].get("eve", []) if not t["rejected"])
    nt_mor = sum(1 for s in valid for t in subj_data_dict[s].get("mor", []) if not t["rejected"])

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_means, mor_means], positions=[1, 2], widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for ev, mo in zip(eve_means, mor_means):
        j = np.random.uniform(-0.06, 0.06)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=60, zorder=4, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=60, zorder=4, linewidths=0.5)

    all_vals = eve_means + mor_means
    y_top = max(all_vals) * 1.08 if max(all_vals) > 0 else max(abs(v) for v in all_vals) * 1.08
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n}, {nt_eve} trials)", f"Morning\n(n={n}, {nt_mor} trials)"],
                       fontsize=10)
    ax.set_ylabel("HR % change from baseline", fontsize=11)
    ax.set_title(f"HR Score: Evening vs Morning\n(N={n} subjects, all conditions)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "boxplot_eve_vs_mor.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Eve vs Mor boxplot: {fname}")


# -- Neg vs Neu boxplot (2 boxes, collapsed across sessions) ------------------

def plot_neg_vs_neu_boxplot(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    neg_means, neu_means, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        neg_sc, neu_sc = [], []
        for sk in ["eve", "mor"]:
            for t in sessions.get(sk, []):
                if t["rejected"] or np.isnan(t["score"]):
                    continue
                (neg_sc if t["label"] == 1 else neu_sc).append(t["score"])
        if neg_sc and neu_sc:
            neg_means.append(np.mean(neg_sc))
            neu_means.append(np.mean(neu_sc))
            valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for Neg vs Neu boxplot"); return

    try:
        _, pval = stats.wilcoxon(neg_means, neu_means); tl = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(neg_means, neu_means); tl = "Paired t"

    n_neg = sum(1 for s in valid for sk in ["eve","mor"]
                for t in subj_data_dict[s].get(sk,[]) if not t["rejected"] and t["label"]==1)
    n_neu = sum(1 for s in valid for sk in ["eve","mor"]
                for t in subj_data_dict[s].get(sk,[]) if not t["rejected"] and t["label"]==2)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([neg_means, neu_means], positions=[1, 2], widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.NEG_COLOR, config.NEU_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for med in bp["medians"]:
        med.set(color="black", linewidth=2)
    for neg_v, neu_v in zip(neg_means, neu_means):
        j = np.random.uniform(-0.06, 0.06)
        lc = config.UP_COLOR if neu_v >= neg_v else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [neg_v, neu_v], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, neg_v, color=config.NEG_COLOR, edgecolors="k", s=60, zorder=4, linewidths=0.5)
        ax.scatter(2+j, neu_v, color=config.NEU_COLOR, edgecolors="k", s=60, zorder=4, linewidths=0.5)

    all_vals = neg_means + neu_means
    y_top = max(all_vals) * 1.08 if max(all_vals) > 0 else max(abs(v) for v in all_vals) * 1.08
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, tl)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Negative\n(n={n}, {n_neg} trials)", f"Neutral\n(n={n}, {n_neu} trials)"],
                       fontsize=10)
    ax.set_ylabel("HR % change from baseline", fontsize=11)
    ax.set_title(f"HR Score: Negative vs Neutral\n(N={n} subjects, Eve + Mor combined)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "boxplot_neg_vs_neu.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Neg vs Neu boxplot: {fname}")


# -- Main ---------------------------------------------------------------------

def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    subjects = [d for d in sorted(os.listdir(config.RAW_DATA_DIR))
                if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))]
    if config.SUBJECT_FILTER:
        subjects = [s for s in subjects if s in config.SUBJECT_FILTER]

    cache          = load_cache()
    sessions_cache = cache.setdefault("sessions", {})
    changed        = False

    for subj in subjects:
        subj_path      = os.path.join(config.RAW_DATA_DIR, subj)
        startle_folder = processor.find_startle_output_folder(subj_path)
        eeg_folder     = os.path.join(subj_path, "EEG")
        if not startle_folder or not os.path.isdir(eeg_folder):
            print(f"[!] Missing data folders for {subj}")
            continue

        print(f"\n{subj}")
        sessions_cache.setdefault(subj, {})

        for sess_key, sess_cfg in config.SESSION_MAP.items():
            if sess_key in sessions_cache[subj] and not config.FORCE_RELOAD:
                n = len(sessions_cache[subj][sess_key].get("trials_meta", []))
                print(f"  {sess_key}: using cache ({n} trials)")
                continue

            csv_path = processor.find_csv_by_suffix(startle_folder, sess_cfg["csv_suffix"])
            mff_path = processor.find_mff_file(eeg_folder, sess_key)
            if not csv_path or not mff_path:
                print(f"  [!] Missing files for {subj} {sess_key}")
                continue

            ratings_df  = processor.load_and_classify_ratings(csv_path)
            session, _  = processor.process_session(
                mff_path, ratings_df, config.HR_CHANNEL, config)
            if session is not None:
                sessions_cache[subj][sess_key] = session
                changed = True

    if changed:
        save_cache({"sessions": sessions_cache})

    print("\nRunning Layer 2 analysis (epoch cutting + baseline + scoring) ...")
    trials_data, traces_cache = processor.apply_analysis_params(sessions_cache, config)

    save_trial_csv(trials_data, config.OUTPUT_DIR)
    plot_raw_sessions(trials_data, traces_cache, config.OUTPUT_DIR)
    plot_session_timecourses(trials_data, config.OUTPUT_DIR)
    plot_trial_scores_per_subject(trials_data, config.OUTPUT_DIR)
    plot_subject_average_timecourse(trials_data, config.OUTPUT_DIR)
    plot_event_windows(trials_data, config.OUTPUT_DIR)
    plot_group_average_timecourse(trials_data, config.OUTPUT_DIR)
    plot_group_boxplot(trials_data, config.OUTPUT_DIR)
    plot_group_ratio_boxplot(trials_data, config.OUTPUT_DIR)
    plot_group_session_comparison(trials_data, config.OUTPUT_DIR)
    plot_group_eve_mor_ratio(trials_data, config.OUTPUT_DIR)
    plot_group_overall_eve_vs_mor(trials_data, config.OUTPUT_DIR)
    plot_group_overall_ratio(trials_data, config.OUTPUT_DIR)
    plot_group_overall_timecourse(trials_data, config.OUTPUT_DIR)
    plot_eve_vs_mor_boxplot(trials_data, config.OUTPUT_DIR)
    plot_neg_vs_neu_boxplot(trials_data, config.OUTPUT_DIR)
    plot_stai_correlations(trials_data, config.OUTPUT_DIR)
    plot_subjective_negative_percentage(config.RAW_DATA_DIR, config.OUTPUT_DIR)
    plot_review_trials(trials_data, config.OUTPUT_DIR)

    print()
    for subj, sessions in sorted(trials_data.items()):
        for sess_key, trials in sessions.items():
            if not trials:
                continue
            n_rej = sum(1 for t in trials if t["rejected"])
            print(f"  {subj} {sess_key}: {len(trials)} trials, {n_rej} rejected")


if __name__ == "__main__":
    main()
