"""
RESP Pipeline — Respiration Analysis
======================================
Mirrors the emg_raw_potentiation.py architecture:
  - Layer-1 cache: raw wide epochs extracted from MFF (process_session)
  - Layer-2 live:  filter → baseline → rejection → score (apply_analysis_params)

Edit resp_config.py (including SUBJECT_PARAMS) and re-run; no MFF reload needed
unless RESP_CHANNELS, WIDE_TMIN, or WIDE_TMAX are changed.

Run from repo root:
    .venv/bin/python -m RESP.resp_main
"""

import os
import pickle
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

import RESP.resp_config as cfg
from RESP.resp_processor import (
    find_startle_output_folder,
    find_mff_file,
    find_csv_by_suffix,
    load_and_classify_ratings,
    process_session,
    apply_analysis_params,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Stat helpers ──────────────────────────────────────────────────────────────

def _sig_label(pval):
    return "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))


def _draw_stat_bracket(ax, x1, x2, y_top, pval, test_label, fontsize=10):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y_top, y_top + h, y_top + h, y_top],
            lw=1.2, color="black")
    sig = _sig_label(pval)
    ax.text((x1 + x2) / 2, y_top + h * 1.5,
            f"{sig}  {test_label} p={pval:.3f}",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def _wilcoxon_or_t(a, b):
    try:
        _, p = stats.wilcoxon(a, b)
        return p, "Wilcoxon"
    except Exception:
        _, p = stats.ttest_rel(a, b)
        return p, "Paired t"


# ── 1. Individual trial plots ─────────────────────────────────────────────────

def plot_individual_trials(subj_data_dict, output_dir):
    """
    One PNG per trial.
    Top block: RSP_Clean for all channels (waveform, one subplot each).
    Bottom block (primary channel only): RSP_Amplitude and RSP_Rate from NeuroKit2.
    Rejected trials shown in grey.  Accepted Negative=red, Neutral=blue.
    """
    base_dir = os.path.join(output_dir, "individual_trials")
    n_rej_total = n_all_total = 0

    for subj, sessions in sorted(subj_data_dict.items()):
        for sess_key in ["mor", "eve"]:
            trials = sessions.get(sess_key)
            if not trials:
                continue
            p = {**cfg.DEFAULT_PARAMS, **cfg.SUBJECT_PARAMS.get(subj, {})}
            sess_dir = os.path.join(base_dir, subj, sess_key)
            os.makedirs(sess_dir, exist_ok=True)

            for i, t in enumerate(trials):
                n_all_total += 1
                rejected = t["rejected"]
                if rejected:
                    n_rej_total += 1
                    line_color = cfg.REJ_COLOR
                    cond_name  = "REJECTED"
                else:
                    line_color = cfg.NEG_COLOR if t["label"] == 1 else cfg.NEU_COLOR
                    cond_name  = "Negative"    if t["label"] == 1 else "Neutral"

                channels  = t["channels"]
                n_ch      = len(channels)
                ref_ch    = cfg.PRIMARY_CHANNEL if cfg.PRIMARY_CHANNEL in channels else channels[0]
                has_nk    = t.get("epoch_amplitude_wide") is not None

                # n_ch clean rows + 2 nk rows (amplitude, rate) for primary channel
                n_rows = n_ch + (2 if has_nk else 0)
                fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2.4 * n_rows), sharex=True)
                if n_rows == 1:
                    axes = [axes]

                tw_ms = t["times_anal"] * 1000 if t["times_anal"] is not None else t["times_wide"] * 1000

                def _add_shading(ax, bm_val=None):
                    ax.axvline(0, color="black", linestyle="--", linewidth=1)
                    ax.axvspan(p["baseline_tmin"] * 1000, p["baseline_tmax"] * 1000,
                               color="gold", alpha=0.15)
                    if not rejected:
                        ax.axvspan(p["score_tmin"] * 1000, p["score_tmax"] * 1000,
                                   color="green", alpha=0.10)
                        if bm_val is not None:
                            ax.axhline(bm_val, color="gray", linestyle=":", linewidth=1.0)
                    ax.grid(True, linestyle="--", alpha=0.35)

                # RSP_Clean rows (all channels)
                clean_src = t.get("epoch_proc_anal") or {}
                for row_i, ch in enumerate(channels):
                    ax  = axes[row_i]
                    sig = clean_src.get(ch, t["epoch_raw_wide"][ch])
                    ax.plot(tw_ms, sig, color=line_color, linewidth=1.2, alpha=0.9)
                    bm_val = (t["baseline_mean"] or {}).get(ch) if p.get("score_signal") == "clean" else None
                    _add_shading(ax, bm_val)
                    ax.set_ylabel(f"{ch}\nClean", fontsize=8)

                # RSP_Amplitude row (primary channel)
                if has_nk:
                    ax_amp = axes[n_ch]
                    amp_wide = t["epoch_amplitude_wide"].get(ref_ch)
                    if amp_wide is not None:
                        anal_mask = (t["times_wide"] >= p["anal_tmin"]) & (t["times_wide"] < p["anal_tmax"])
                        ax_amp.plot(tw_ms, amp_wide[anal_mask],
                                    color=line_color, linewidth=1.4, alpha=0.9)
                        bm_val = (t["baseline_mean"] or {}).get(ref_ch) if p.get("score_signal") == "amplitude" else None
                        _add_shading(ax_amp, bm_val)
                    ax_amp.set_ylabel(f"{ref_ch}\nAmplitude", fontsize=8)

                    # RSP_Rate row (primary channel)
                    ax_rate = axes[n_ch + 1]
                    rate_wide = t["epoch_rate_wide"].get(ref_ch)
                    if rate_wide is not None:
                        ax_rate.plot(tw_ms, rate_wide[anal_mask],
                                     color=line_color, linewidth=1.4, alpha=0.9)
                        bm_val = (t["baseline_mean"] or {}).get(ref_ch) if p.get("score_signal") == "rate" else None
                        _add_shading(ax_rate, bm_val)
                    ax_rate.set_ylabel(f"{ref_ch}\nRate (bpm)", fontsize=8)

                axes[-1].set_xlabel("Time (ms relative to startle)", fontsize=9)

                phase_str  = ("inhale" if t.get("phase_at_onset") == 1 else
                              "exhale" if t.get("phase_at_onset") == 0 else "?")
                score_str  = (f"{t['score']:.4f}" if not rejected and t["score"] is not None
                              else "REJECTED")
                sig_label  = p.get("score_signal", "amplitude")
                fig.suptitle(
                    f"{subj} | {sess_key.upper()} | Trial {i+1} | {cond_name}  "
                    f"[phase@onset: {phase_str}]\n"
                    f"Image: {t['image_detail']} | Val: {t['valence']} | "
                    f"Ar: {t['arousal']} | Score ({ref_ch} {sig_label}): {score_str}",
                    fontsize=10, fontweight="bold",
                )
                plt.tight_layout()
                fname = os.path.join(sess_dir, f"trial_{i+1:03d}_{cond_name[:3].upper()}.png")
                fig.savefig(fname, dpi=120, bbox_inches="tight")
                plt.close(fig)

    pct = 100 * n_rej_total / max(n_all_total, 1)
    print(f"  Individual trials: {n_all_total} total, {n_rej_total} rejected ({pct:.0f}%)")


# ── 2. Trial-score scatter per subject ────────────────────────────────────────

def plot_trial_scores_per_subject(subj_data_dict, output_dir):
    plot_dir = os.path.join(output_dir, "trial_score_scatter")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(f"{subj} – Respiration Score by Trial Number ({cfg.PRIMARY_CHANNEL})",
                     fontsize=13, fontweight="bold")

        for ax, sess_key in zip(axes, ["mor", "eve"]):
            trials = sessions.get(sess_key)
            if not trials:
                ax.set_title(f"{sess_key.upper()} – no data")
                continue

            scores     = np.array([t["score"] for t in trials], dtype=float)
            labels     = np.array([t["label"]    for t in trials])
            rejected   = np.array([t["rejected"] for t in trials])
            trial_nums = np.arange(1, len(trials) + 1)

            for cond, color, name in [(1, cfg.NEG_COLOR, "Negative"), (2, cfg.NEU_COLOR, "Neutral")]:
                mask = (~rejected) & (labels == cond)
                if mask.sum():
                    ax.scatter(trial_nums[mask], scores[mask], color=color,
                               s=60, edgecolors="white", linewidths=0.5, zorder=3,
                               label=f"{name} (n={mask.sum()})")

            n_rej = rejected.sum()
            if n_rej:
                ax.scatter(trial_nums[rejected], np.zeros(n_rej),
                           color=cfg.REJ_COLOR, marker="x", s=80, zorder=2,
                           label=f"Rejected (n={n_rej})")

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(cfg.SESSION_MAP[sess_key]["label"], fontsize=11, fontweight="bold")
            ax.set_xlabel("Trial number")
            if ax is axes[0]:
                ax.set_ylabel("Score (mean − baseline, signal units)")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{subj}_trial_scores.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved trial-score scatter: {fname}")


# ── 3. Group boxplot — four conditions ───────────────────────────────────────

def plot_group_boxplot_four_conditions(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cond_keys = [
        ("eve", 1, "Eve-Neg", cfg.NEG_COLOR,  "#E8A99A"),
        ("eve", 2, "Eve-Neu", cfg.NEU_COLOR,  "#9AC4E8"),
        ("mor", 1, "Mor-Neg", "#922B21",      "#E8A99A"),
        ("mor", 2, "Mor-Neu", "#1A5276",      "#9AC4E8"),
    ]

    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sess_key, cond, *_ in cond_keys:
            trials = sessions.get(sess_key, [])
            vals   = [t["score"] for t in trials if not t["rejected"] and t["label"] == cond]
            if vals:
                subj_means[subj][(sess_key, cond)] = float(np.mean(vals))

    valid = [s for s in subj_means if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    if len(valid) < 2:
        print("  [!] Not enough subjects for four-condition boxplot")
        return

    data_cols = [[subj_means[s][(sk, c)] for s in valid] for sk, c, *_ in cond_keys]
    n_trials  = []
    for sk, c, *_ in cond_keys:
        n_trials.append(sum(1 for s in valid
                            for t in subj_data_dict[s].get(sk, [])
                            if not t["rejected"] and t["label"] == c))

    fig, ax = plt.subplots(figsize=(10, 6))
    positions  = [1, 2, 3, 4]
    box_colors = [ck[4] for ck in cond_keys]
    bp = ax.boxplot(data_cols, positions=positions, widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.7); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for s in valid:
        j = np.random.uniform(-0.07, 0.07)
        v_en = subj_means[s].get(("eve", 1))
        v_eu = subj_means[s].get(("eve", 2))
        if v_en is not None and v_eu is not None:
            lc = cfg.UP_COLOR if v_eu >= v_en else cfg.DOWN_COLOR
            ax.plot([1+j, 2+j], [v_en, v_eu], color=lc, alpha=0.45, linewidth=1.2)
            ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(2+j, v_eu, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        v_mn = subj_means[s].get(("mor", 1))
        v_mu = subj_means[s].get(("mor", 2))
        if v_mn is not None and v_mu is not None:
            lc = cfg.UP_COLOR if v_mu >= v_mn else cfg.DOWN_COLOR
            ax.plot([3+j, 4+j], [v_mn, v_mu], color=lc, alpha=0.45, linewidth=1.2)
            ax.scatter(3+j, v_mn, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(4+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    eve_neg, eve_neu = data_cols[0], data_cols[1]
    mor_neg, mor_neu = data_cols[2], data_cols[3]
    p_eve, tl_eve = _wilcoxon_or_t(eve_neg, eve_neu)
    p_mor, tl_mor = _wilcoxon_or_t(mor_neg, mor_neu)

    flat = [v for col in data_cols for v in col]
    y_top_eve = max(max(eve_neg), max(eve_neu)) * 1.10
    y_top_mor = max(max(mor_neg), max(mor_neu)) * 1.10
    ax.set_ylim(top=max(y_top_eve, y_top_mor) * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top_eve, p_eve, tl_eve)
    _draw_stat_bracket(ax, 3, 4, y_top_mor, p_mor, tl_mor)

    tick_labels = [f"{ck[2]}\n(n={nt})" for ck, nt in zip(cond_keys, n_trials)]
    ax.set_xticks(positions); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel(f"Resp Score — {cfg.PRIMARY_CHANNEL} (mean − baseline)", fontsize=11)
    ax.set_title(f"Group Respiration Scores by Condition (N={len(valid)} subjects)\n"
                 f"Lines connect Neg↔Neu within session per subject",
                 fontsize=12, fontweight="bold")
    ax.axvline(2.5, color="lightgray", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_four_conditions_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved four-condition boxplot: {fname}")


# ── 4. Group ratio boxplot (Neg/Neu per session) ─────────────────────────────

def plot_group_ratio_boxplot(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    eve_ratios, mor_ratios, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        ratios = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            neg = [t["score"] for t in trials if not t["rejected"] and t["label"] == 1]
            neu = [t["score"] for t in trials if not t["rejected"] and t["label"] == 2]
            if neg and neu and np.mean(neu) != 0:
                ratios[sess_key] = float(np.mean(neg)) / float(np.mean(neu))
        if "eve" in ratios and "mor" in ratios:
            eve_ratios.append(ratios["eve"])
            mor_ratios.append(ratios["mor"])
            valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for ratio boxplot")
        return

    pval, test_label = _wilcoxon_or_t(eve_ratios, mor_ratios)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_ratios, mor_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [cfg.EVE_COLOR, cfg.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for ev, mo in zip(eve_ratios, mor_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = cfg.UP_COLOR if mo >= ev else cfg.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=cfg.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=cfg.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_ratios), max(mor_ratios)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n})", f"Morning\n(n={n})"], fontsize=11)
    ax.set_ylabel(f"Neg / Neu Ratio — {cfg.PRIMARY_CHANNEL}", fontsize=11)
    ax.set_title(f"Neg-to-Neutral Ratio: Evening vs Morning (N={n} subjects)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_neg_neu_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Neg/Neu ratio boxplot: {fname}")


# ── 5. Group average timecourse — primary channel ────────────────────────────

def plot_group_average_timecourse(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"Group Average Respiration Timecourse — {cfg.PRIMARY_CHANNEL} "
        f"(accepted trials, baseline-corrected)",
        fontsize=13, fontweight="bold",
    )
    p_global = cfg.DEFAULT_PARAMS

    for ax, sess_key in zip(axes, ["eve", "mor"]):
        sess_label = cfg.SESSION_MAP[sess_key]["label"]
        neg_epochs, neu_epochs = [], []
        n_neg_subj = n_neu_subj = 0

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            if not trials:
                continue
            bm = lambda t: (t["baseline_mean"] or {}).get(cfg.PRIMARY_CHANNEL, 0.0)
            subj_neg = [(t["epoch_proc_anal"][cfg.PRIMARY_CHANNEL] - bm(t))
                        for t in trials if not t["rejected"] and t["label"] == 1
                        and cfg.PRIMARY_CHANNEL in (t["epoch_proc_anal"] or {})]
            subj_neu = [(t["epoch_proc_anal"][cfg.PRIMARY_CHANNEL] - bm(t))
                        for t in trials if not t["rejected"] and t["label"] == 2
                        and cfg.PRIMARY_CHANNEL in (t["epoch_proc_anal"] or {})]
            if subj_neg:
                neg_epochs.append(np.mean(subj_neg, axis=0))
                n_neg_subj += 1
            if subj_neu:
                neu_epochs.append(np.mean(subj_neu, axis=0))
                n_neu_subj += 1

        times_ms = None
        for _, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            if trials and trials[0].get("times_anal") is not None:
                times_ms = trials[0]["times_anal"] * 1000
                break

        if times_ms is None or (not neg_epochs and not neu_epochs):
            ax.set_title(f"{sess_label} – no data")
            continue

        for epochs, color, name, n_subj in [
            (neg_epochs, cfg.NEG_COLOR, "Negative", n_neg_subj),
            (neu_epochs, cfg.NEU_COLOR, "Neutral",  n_neu_subj),
        ]:
            if not epochs:
                continue
            arr = np.array(epochs)
            avg = np.mean(arr, axis=0)
            sem = stats.sem(arr, axis=0, nan_policy="omit")
            n_trials_total = sum(
                len([t for t in sess.get(sess_key, [])
                     if not t["rejected"] and t["label"] == (1 if name == "Negative" else 2)])
                for _, sess in subj_data_dict.items()
            )
            ax.plot(times_ms, avg, color=color, linewidth=2.2,
                    label=f"{name} (n subj={n_subj}, n trials={n_trials_total})")
            ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
        ax.axvspan(p_global["score_tmin"] * 1000, p_global["score_tmax"] * 1000,
                   color="green", alpha=0.08, label="Score window")
        ax.set_title(sess_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Amplitude (baseline-corrected, a.u.)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(output_dir, "group_average_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group average timecourse: {fname}")


# ── 6. Per-subject average timecourse — primary channel ──────────────────────

def plot_subject_average_timecourse(subj_data_dict, output_dir):
    plot_dir = os.path.join(output_dir, "subject_average_timecourses")
    os.makedirs(plot_dir, exist_ok=True)
    p_global = cfg.DEFAULT_PARAMS

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(
            f"{subj} – Average Respiration Timecourse — {cfg.PRIMARY_CHANNEL}",
            fontsize=13, fontweight="bold",
        )
        has_data = False

        for ax, sess_key in zip(axes, ["eve", "mor"]):
            trials = sessions.get(sess_key, [])
            if not trials:
                ax.set_title(f"{cfg.SESSION_MAP[sess_key]['label']} – no data")
                continue

            times_ms = trials[0]["times_anal"] * 1000

            for cond, color, name in [(1, cfg.NEG_COLOR, "Negative"), (2, cfg.NEU_COLOR, "Neutral")]:
                epochs = [
                    (t["epoch_proc_anal"][cfg.PRIMARY_CHANNEL]
                     - (t["baseline_mean"] or {}).get(cfg.PRIMARY_CHANNEL, 0.0))
                    for t in trials
                    if not t["rejected"] and t["label"] == cond
                    and cfg.PRIMARY_CHANNEL in (t["epoch_proc_anal"] or {})
                ]
                if not epochs:
                    continue
                has_data = True
                arr = np.array(epochs)
                avg = np.mean(arr, axis=0)
                sem = (stats.sem(arr, axis=0, nan_policy="omit")
                       if len(epochs) > 1 else np.zeros_like(avg))
                ax.plot(times_ms, avg, color=color, linewidth=2.2,
                        label=f"{name} (n={len(epochs)})")
                if len(epochs) > 1:
                    ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

            ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
            ax.axvspan(p_global["baseline_tmin"] * 1000, p_global["baseline_tmax"] * 1000,
                       color="gold", alpha=0.12, label="Baseline")
            ax.axvspan(p_global["score_tmin"] * 1000, p_global["score_tmax"] * 1000,
                       color="green", alpha=0.08, label="Score window")
            ax.set_title(cfg.SESSION_MAP[sess_key]["label"], fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("Amplitude (baseline-corrected, a.u.)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        if has_data:
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_average_timecourse.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved subject average timecourse: {fname}")
        plt.close(fig)


# ── 7. Multi-channel group timecourse (RESP-specific) ────────────────────────

def plot_multichannel_group_timecourse(subj_data_dict, output_dir):
    """
    Grid: rows = RESP channels, cols = sessions (Eve | Mor).
    Each cell shows grand-average ± SEM for Neg (red) and Neu (blue).
    This is the primary respiration diagnostic plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    p_global = cfg.DEFAULT_PARAMS

    all_channels = cfg.RESP_CHANNELS
    n_ch   = len(all_channels)
    n_sess = 2
    fig, axes = plt.subplots(n_ch, n_sess, figsize=(13, 3.2 * n_ch), sharey=False)
    if n_ch == 1:
        axes = [axes]
    fig.suptitle("Multi-Channel Group Respiration Timecourse (baseline-corrected, ±SEM)",
                 fontsize=13, fontweight="bold")

    for row, ch in enumerate(all_channels):
        ch_color = cfg.CHANNEL_COLORS.get(ch, "#555555")
        for col, sess_key in enumerate(["eve", "mor"]):
            ax = axes[row][col]
            sess_label = cfg.SESSION_MAP[sess_key]["label"]
            neg_epochs, neu_epochs = [], []
            n_neg_s = n_neu_s = 0

            for subj, sessions in subj_data_dict.items():
                trials = sessions.get(sess_key, [])
                if not trials:
                    continue
                bm = lambda t: (t["baseline_mean"] or {}).get(ch, 0.0)
                subj_neg = [
                    t["epoch_proc_anal"][ch] - bm(t)
                    for t in trials
                    if not t["rejected"] and t["label"] == 1 and ch in (t["epoch_proc_anal"] or {})
                ]
                subj_neu = [
                    t["epoch_proc_anal"][ch] - bm(t)
                    for t in trials
                    if not t["rejected"] and t["label"] == 2 and ch in (t["epoch_proc_anal"] or {})
                ]
                if subj_neg:
                    neg_epochs.append(np.mean(subj_neg, axis=0)); n_neg_s += 1
                if subj_neu:
                    neu_epochs.append(np.mean(subj_neu, axis=0)); n_neu_s += 1

            times_ms = None
            for _, sessions in subj_data_dict.items():
                trials = sessions.get(sess_key, [])
                if trials and trials[0].get("times_anal") is not None:
                    times_ms = trials[0]["times_anal"] * 1000
                    break

            if times_ms is None:
                ax.set_title(f"{ch} | {sess_label} – no data", fontsize=9)
                continue

            for epochs, cond_color, cond_name, n_s in [
                (neg_epochs, cfg.NEG_COLOR, "Neg", n_neg_s),
                (neu_epochs, cfg.NEU_COLOR, "Neu", n_neu_s),
            ]:
                if not epochs:
                    continue
                arr = np.array(epochs)
                avg = np.mean(arr, axis=0)
                sem = stats.sem(arr, axis=0, nan_policy="omit")
                ax.plot(times_ms, avg, color=cond_color, linewidth=2.0,
                        label=f"{cond_name} (n={n_s})")
                ax.fill_between(times_ms, avg - sem, avg + sem,
                                color=cond_color, alpha=0.18)

            ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
            ax.axvspan(p_global["score_tmin"] * 1000, p_global["score_tmax"] * 1000,
                       color="green", alpha=0.07)
            if row == 0:
                ax.set_title(sess_label, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{ch}\n(a.u.)", color=ch_color, fontsize=9, fontweight="bold")
            if row == n_ch - 1:
                ax.set_xlabel("Time (ms)", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.85, loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    fname = os.path.join(output_dir, "multichannel_group_timecourse.png")
    fig.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved multi-channel group timecourse: {fname}")


# ── 8. Per-subject multi-channel timecourse ───────────────────────────────────

def plot_subject_multichannel_timecourse(subj_data_dict, output_dir):
    """
    Same grid as plot_multichannel_group_timecourse but for one subject at a time.
    """
    plot_dir = os.path.join(output_dir, "subject_multichannel_timecourses")
    os.makedirs(plot_dir, exist_ok=True)
    p_global = cfg.DEFAULT_PARAMS

    for subj, sessions in sorted(subj_data_dict.items()):
        all_channels = cfg.RESP_CHANNELS
        n_ch = len(all_channels)
        fig, axes = plt.subplots(n_ch, 2, figsize=(13, 3.0 * n_ch), sharey=False)
        if n_ch == 1:
            axes = [axes]
        fig.suptitle(f"{subj} – Multi-Channel Respiration Timecourse (baseline-corrected)",
                     fontsize=12, fontweight="bold")
        has_data = False

        for row, ch in enumerate(all_channels):
            ch_color = cfg.CHANNEL_COLORS.get(ch, "#555555")
            for col, sess_key in enumerate(["eve", "mor"]):
                ax = axes[row][col]
                trials = sessions.get(sess_key, [])
                if not trials:
                    ax.set_title(f"{ch} | {cfg.SESSION_MAP[sess_key]['label']} – no data", fontsize=8)
                    continue

                times_ms = trials[0]["times_anal"] * 1000

                for cond, color, name in [(1, cfg.NEG_COLOR, "Neg"), (2, cfg.NEU_COLOR, "Neu")]:
                    epochs = [
                        t["epoch_proc_anal"][ch] - (t["baseline_mean"] or {}).get(ch, 0.0)
                        for t in trials
                        if not t["rejected"] and t["label"] == cond and ch in (t["epoch_proc_anal"] or {})
                    ]
                    if not epochs:
                        continue
                    has_data = True
                    arr = np.array(epochs)
                    avg = np.mean(arr, axis=0)
                    sem = (stats.sem(arr, axis=0, nan_policy="omit")
                           if len(epochs) > 1 else np.zeros_like(avg))
                    ax.plot(times_ms, avg, color=color, linewidth=2.0,
                            label=f"{name} (n={len(epochs)})")
                    if len(epochs) > 1:
                        ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.18)

                ax.axvline(0, color="k", linestyle="--", linewidth=1.0)
                ax.axvspan(p_global["score_tmin"] * 1000, p_global["score_tmax"] * 1000,
                           color="green", alpha=0.08)
                if row == 0:
                    ax.set_title(cfg.SESSION_MAP[sess_key]["label"], fontsize=10, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(ch, color=ch_color, fontsize=9, fontweight="bold")
                if row == n_ch - 1:
                    ax.set_xlabel("Time (ms)", fontsize=9)
                ax.legend(fontsize=8, framealpha=0.85)
                ax.grid(True, linestyle="--", alpha=0.35)

        if has_data:
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_multichannel_timecourse.png")
            fig.savefig(fname, dpi=140, bbox_inches="tight")
            print(f"  Saved subject multichannel timecourse: {fname}")
        plt.close(fig)


# ── 9. Group session comparison (Eve-Neg/Mor-Neg/Eve-Neu/Mor-Neu) ────────────

def plot_group_session_comparison(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cond_keys = [
        ("eve", 1, "Eve-Neg", cfg.NEG_COLOR),
        ("mor", 1, "Mor-Neg", "#922B21"),
        ("eve", 2, "Eve-Neu", cfg.NEU_COLOR),
        ("mor", 2, "Mor-Neu", "#1A5276"),
    ]

    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sk, c, *_ in cond_keys:
            vals = [t["score"] for t in sessions.get(sk, []) if not t["rejected"] and t["label"] == c]
            if vals:
                subj_means[subj][(sk, c)] = float(np.mean(vals))

    valid = [s for s in subj_means if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    if len(valid) < 2:
        print("  [!] Not enough subjects for session-comparison boxplot")
        return

    data_cols = [[subj_means[s][(sk, c)] for s in valid] for sk, c, *_ in cond_keys]
    n_trials  = [sum(1 for s in valid
                     for t in subj_data_dict[s].get(sk, []) if not t["rejected"] and t["label"] == c)
                 for sk, c, *_ in cond_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions  = [1, 2, 3.6, 4.6]
    box_colors = [ck[3] for ck in cond_keys]
    bp = ax.boxplot(data_cols, positions=positions, widths=0.42,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for s in valid:
        j = np.random.uniform(-0.07, 0.07)
        v_en = subj_means[s][("eve", 1)]; v_mn = subj_means[s][("mor", 1)]
        lc = cfg.UP_COLOR if v_mn >= v_en else cfg.DOWN_COLOR
        ax.plot([1+j, 2+j], [v_en, v_mn], color=lc, alpha=0.55, linewidth=1.4)
        ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(2+j, v_mn, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        v_eu = subj_means[s][("eve", 2)]; v_mu = subj_means[s][("mor", 2)]
        lc = cfg.UP_COLOR if v_mu >= v_eu else cfg.DOWN_COLOR
        ax.plot([3.6+j, 4.6+j], [v_eu, v_mu], color=lc, alpha=0.55, linewidth=1.4)
        ax.scatter(3.6+j, v_eu, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(4.6+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    p_neg, tl_neg = _wilcoxon_or_t(data_cols[0], data_cols[1])
    p_neu, tl_neu = _wilcoxon_or_t(data_cols[2], data_cols[3])
    y_top_neg = max(max(data_cols[0]), max(data_cols[1])) * 1.10
    y_top_neu = max(max(data_cols[2]), max(data_cols[3])) * 1.10
    ax.set_ylim(top=max(y_top_neg, y_top_neu) * 1.20)
    _draw_stat_bracket(ax, 1,   2,   y_top_neg, p_neg, tl_neg)
    _draw_stat_bracket(ax, 3.6, 4.6, y_top_neu, p_neu, tl_neu)

    tick_labels = [f"{ck[2]}\n(n={nt})" for ck, nt in zip(cond_keys, n_trials)]
    ax.set_xticks(positions); ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel(f"Resp Score — {cfg.PRIMARY_CHANNEL} (mean − baseline)", fontsize=11)
    ax.set_title(f"Evening vs Morning by Condition (N={len(valid)} subjects)\n"
                 f"Green = score increased, Red = decreased",
                 fontsize=12, fontweight="bold")
    ax.axvline(2.8, color="lightgray", linestyle="--", linewidth=1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_session_comparison_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved session-comparison boxplot: {fname}")


# ── 10. Eve/Mor ratio per condition ──────────────────────────────────────────

def plot_group_eve_mor_ratio(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    neg_ratios, neu_ratios, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        r = {}
        for sk in ["eve", "mor"]:
            for c in [1, 2]:
                vals = [t["score"] for t in sessions.get(sk, []) if not t["rejected"] and t["label"] == c]
                if vals:
                    r[(sk, c)] = float(np.mean(vals))
        if all(k in r for k in [("eve",1),("mor",1),("eve",2),("mor",2)]):
            if r[("mor",1)] != 0 and r[("mor",2)] != 0:
                neg_ratios.append(r[("eve",1)] / r[("mor",1)])
                neu_ratios.append(r[("eve",2)] / r[("mor",2)])
                valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for Eve/Mor ratio plot")
        return

    pval, test_label = _wilcoxon_or_t(neg_ratios, neu_ratios)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([neg_ratios, neu_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [cfg.NEG_COLOR, cfg.NEU_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    for nr, neu_r in zip(neg_ratios, neu_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = cfg.UP_COLOR if neu_r >= nr else cfg.DOWN_COLOR
        ax.plot([1+j, 2+j], [nr, neu_r], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, nr,   color=cfg.NEG_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, neu_r, color=cfg.NEU_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(neg_ratios), max(neu_ratios)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Negative\n(n={n})", f"Neutral\n(n={n})"], fontsize=11)
    ax.set_ylabel(f"Eve / Mor Score Ratio — {cfg.PRIMARY_CHANNEL}", fontsize=11)
    ax.set_title(f"Eve-to-Mor Ratio: Neg vs Neu (N={n} subjects)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_eve_mor_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Eve/Mor ratio boxplot: {fname}")


# ── 11. Overall Eve vs Mor (all trials) ──────────────────────────────────────

def plot_group_overall_eve_vs_mor(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    eve_means, mor_means, valid = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sk in ["eve", "mor"]:
            scores = [t["score"] for t in sessions.get(sk, []) if not t["rejected"]]
            if scores:
                vals[sk] = float(np.mean(scores))
        if "eve" in vals and "mor" in vals:
            eve_means.append(vals["eve"])
            mor_means.append(vals["mor"])
            valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for overall Eve vs Mor plot")
        return

    pval, test_label = _wilcoxon_or_t(eve_means, mor_means)
    n_eve = sum(sum(1 for t in subj_data_dict[s].get("eve",[]) if not t["rejected"]) for s in valid)
    n_mor = sum(sum(1 for t in subj_data_dict[s].get("mor",[]) if not t["rejected"]) for s in valid)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_means, mor_means], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [cfg.EVE_COLOR, cfg.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    for ev, mo in zip(eve_means, mor_means):
        j = np.random.uniform(-0.05, 0.05)
        lc = cfg.UP_COLOR if mo >= ev else cfg.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, ev, color=cfg.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=cfg.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(eve_means), max(mor_means)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n subj={n}, n trials={n_eve})",
                        f"Morning\n(n subj={n}, n trials={n_mor})"], fontsize=10)
    ax.set_ylabel(f"Resp Score — {cfg.PRIMARY_CHANNEL}", fontsize=11)
    ax.set_title(f"Overall Score: Evening vs Morning (N={n} subjects, all accepted trials)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_eve_vs_mor.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall Eve vs Mor: {fname}")


# ── 12. Overall Eve/Mor ratio (vs 1) ─────────────────────────────────────────

def plot_group_overall_ratio(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ratios, valid = [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sk in ["eve", "mor"]:
            scores = [t["score"] for t in sessions.get(sk, []) if not t["rejected"]]
            if scores:
                vals[sk] = float(np.mean(scores))
        if "eve" in vals and "mor" in vals and vals["mor"] != 0:
            ratios.append(vals["eve"] / vals["mor"])
            valid.append(subj)

    n = len(valid)
    if n < 2:
        print("  [!] Not enough subjects for overall ratio plot")
        return

    try:
        _, pval = stats.wilcoxon([r - 1 for r in ratios]); test_label = "Wilcoxon (vs 1)"
    except Exception:
        _, pval = stats.ttest_1samp(ratios, 1);           test_label = "One-sample t (vs 1)"

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(ratios, positions=[1], widths=0.35, patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#95A5A6"); bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][0].set_edgecolor("gray")
    bp["medians"][0].set(color="black", linewidth=2)
    for r in ratios:
        j = np.random.uniform(-0.08, 0.08)
        ax.scatter(1+j, r, color="#7F8C8D", edgecolors="k", s=60, zorder=3, linewidths=0.5)
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.5)
    y_top = max(ratios) * 1.08
    ax.set_ylim(top=y_top * 1.20)
    sig = _sig_label(pval)
    ax.text(1, y_top * 1.05, f"{sig}\n{test_label}\np={pval:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks([1]); ax.set_xticklabels([f"Eve / Mor\n(n={n})"], fontsize=11)
    ax.set_ylabel(f"Eve / Mor Score Ratio — {cfg.PRIMARY_CHANNEL}", fontsize=11)
    ax.set_title("Overall Eve/Mor Ratio (all accepted trials, tested vs ratio=1)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_ratio.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall ratio: {fname}")


# ── 13. Overall average timecourse (Eve vs Mor, all trials) ──────────────────

def plot_group_overall_timecourse(subj_data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    p_global = cfg.DEFAULT_PARAMS
    fig, ax = plt.subplots(figsize=(9, 5))
    times_ms = None

    for sess_key, sess_label, color in [("eve","Evening",cfg.EVE_COLOR),("mor","Morning",cfg.MOR_COLOR)]:
        subj_avgs = []
        n_trials_total = 0
        for subj, sessions in subj_data_dict.items():
            trials = [t for t in sessions.get(sess_key, [])
                      if not t["rejected"] and cfg.PRIMARY_CHANNEL in (t["epoch_proc_anal"] or {})]
            if not trials:
                continue
            epochs = [(t["epoch_proc_anal"][cfg.PRIMARY_CHANNEL]
                       - (t["baseline_mean"] or {}).get(cfg.PRIMARY_CHANNEL, 0.0))
                      for t in trials]
            subj_avgs.append(np.mean(epochs, axis=0))
            n_trials_total += len(trials)
            if times_ms is None:
                times_ms = trials[0]["times_anal"] * 1000

        if not subj_avgs or times_ms is None:
            continue
        arr = np.array(subj_avgs)
        avg = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0, nan_policy="omit")
        ax.plot(times_ms, avg, color=color, linewidth=2.5,
                label=f"{sess_label} (n subj={len(subj_avgs)}, n trials={n_trials_total})")
        ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
    ax.axvspan(p_global["score_tmin"]*1000, p_global["score_tmax"]*1000,
               color="green", alpha=0.08, label="Score window")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel(f"Amplitude — {cfg.PRIMARY_CHANNEL} (baseline-corrected, a.u.)", fontsize=11)
    ax.set_title("Overall Average Respiration Timecourse: Evening vs Morning (±SEM)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fname = os.path.join(output_dir, "group_overall_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall timecourse: {fname}")


# ── 14. STAI-T correlations ───────────────────────────────────────────────────

def plot_stai_correlations(subj_data_dict, output_dir, xlsx_path=None):
    if xlsx_path is None:
        xlsx_path = cfg.SUBJECTS_XLSX
    os.makedirs(output_dir, exist_ok=True)

    def safe_ratio(a, b):
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)

    def safe_mean(lst):
        valid = [x for x in lst if not pd.isna(x) and not np.isinf(x)]
        return float(np.mean(valid)) if valid else np.nan

    try:
        stai_df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [!] Cannot read {xlsx_path}: {e}")
        return

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
            vals = [t["score"] for t in sessions.get(sk,[]) if not t["rejected"] and t["label"] == cv]
            m[f"resp_{sk}_{cl}"] = float(np.mean(vals)) if vals else np.nan

        m["resp_neg_neu_ratio"] = safe_mean([
            safe_ratio(m.get("resp_eve_neg"), m.get("resp_eve_neu")),
            safe_ratio(m.get("resp_mor_neg"), m.get("resp_mor_neu")),
        ])
        m["resp_eve_mor_ratio"] = safe_mean([
            safe_ratio(m.get("resp_eve_neg"), m.get("resp_mor_neg")),
            safe_ratio(m.get("resp_eve_neu"), m.get("resp_mor_neu")),
        ])

        for sk, lbl in [("eve","eve"),("mor","mor")]:
            neg_t = [t for t in sessions.get(sk,[]) if not t["rejected"] and t["label"] == 1]
            m[f"val_{lbl}"]  = float(np.mean([t["valence"] for t in neg_t])) if neg_t else np.nan
            m[f"arsl_{lbl}"] = float(np.mean([t["arousal"] for t in neg_t])) if neg_t else np.nan

        m["val_ratio"]  = safe_ratio(m.get("val_eve"),  m.get("val_mor"))
        m["arsl_ratio"] = safe_ratio(m.get("arsl_eve"), m.get("arsl_mor"))

        for sk in ["eve","mor"]:
            trials = sessions.get(sk,[])
            m[f"neg_pct_{sk}"] = (100.0 * sum(1 for t in trials if t["label"]==1) / len(trials)
                                  if trials else np.nan)
        rows.append(m)

    if len(rows) < 3:
        print("  [!] Not enough subjects with STAI-T data")
        return

    df = pd.DataFrame(rows).set_index("subj")
    panels = [
        ("resp_eve_neg",       "Resp Score (a.u.)",  f"Evening – Negative ({cfg.PRIMARY_CHANNEL})"),
        ("resp_eve_neu",       "Resp Score (a.u.)",  f"Evening – Neutral ({cfg.PRIMARY_CHANNEL})"),
        ("resp_mor_neg",       "Resp Score (a.u.)",  f"Morning – Negative ({cfg.PRIMARY_CHANNEL})"),
        ("resp_mor_neu",       "Resp Score (a.u.)",  f"Morning – Neutral ({cfg.PRIMARY_CHANNEL})"),
        ("resp_neg_neu_ratio", "Neg / Neu Ratio",    "Neg/Neu Ratio (avg sessions)"),
        ("resp_eve_mor_ratio", "Eve / Mor Ratio",    "Eve/Mor Ratio (avg conditions)"),
        ("val_eve",            "Valence rating",     "Valence – Evening (Neg trials)"),
        ("val_mor",            "Valence rating",     "Valence – Morning (Neg trials)"),
        ("val_ratio",          "Valence Ratio",      "Valence Ratio Eve/Mor"),
        ("arsl_eve",           "Arousal rating",     "Arousal – Evening (Neg trials)"),
        ("arsl_mor",           "Arousal rating",     "Arousal – Morning (Neg trials)"),
        ("arsl_ratio",         "Arousal Ratio",      "Arousal Ratio Eve/Mor"),
        ("neg_pct_eve",        "% Neg trials",       "% Neg trials (Evening)"),
    ]

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axes = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(panels):
        ax = axes[idx]
        if col not in df.columns:
            ax.axis("off"); continue
        y_vals = df[col].values.astype(float)
        x_vals = df["stai"].values.astype(float)
        mask   = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x, y   = x_vals[mask], y_vals[mask]
        n      = mask.sum()
        if n < 3 or np.std(x) == 0 or np.std(y) == 0:
            ax.set_title(title + "\n(no variance)", fontsize=10); ax.axis("off"); continue

        r, p = stats.pearsonr(x, y)
        sig  = _sig_label(p)
        dot_color = (cfg.NEG_COLOR if "neg" in col.lower() else
                     cfg.NEU_COLOR if "neu" in col.lower() else "#7F8C8D")
        ax.scatter(x, y, color=dot_color, edgecolors="white", s=60,
                   linewidths=0.5, zorder=3, alpha=0.85)
        m_c, b_c = np.polyfit(x, y, 1)
        x_line   = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m_c * x_line + b_c, color="#2C3E50",
                linewidth=1.6, linestyle="--", zorder=2)
        for sid, xi, yi in zip(df.index[mask], x, y):
            ax.annotate(sid, (xi, yi), fontsize=6, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points", color="#555555")
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(0.04, 0.97, f"r={r:+.2f}  {p_str}  {sig}\nn={n}",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85))
        ax.set_xlabel("STAI-T score", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(f"STAI-T Correlations — {cfg.PRIMARY_CHANNEL} Respiration & Subjective Ratings",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "stai_correlations.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved STAI-T correlations: {fname}")


# ── 15. Subjective Negative % (CSV-only) ─────────────────────────────────────

def plot_subjective_negative_percentage(raw_data_dir, output_dir):
    subject_folders = sorted([
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ])
    data = []
    for subj in subject_folders:
        subj_path     = os.path.join(raw_data_dir, subj)
        startle_folder = find_startle_output_folder(subj_path)
        if not startle_folder:
            continue
        pcts = {}
        for sk, info in cfg.SESSION_MAP.items():
            csv_path = find_csv_by_suffix(startle_folder, info["csv_suffix"])
            if not csv_path:
                continue
            try:
                df = load_and_classify_ratings(csv_path)
                if len(df):
                    pcts[sk] = 100.0 * sum(df["subjective_label"] == 1) / len(df)
            except Exception as e:
                print(f"  [!] Ratings error {subj} {sk}: {e}")
        if "eve" in pcts and "mor" in pcts:
            data.append({"subj": subj, "eve_pct": pcts["eve"], "mor_pct": pcts["mor"]})

    n = len(data)
    if n < 2:
        print("  [!] Not enough subjects for negative% plot")
        return

    df_pcts  = pd.DataFrame(data)
    eve_vals = df_pcts["eve_pct"].values
    mor_vals = df_pcts["mor_pct"].values
    pval, test_label = _wilcoxon_or_t(eve_vals, mor_vals)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_vals, mor_vals], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [cfg.EVE_COLOR, cfg.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    for _, row in df_pcts.iterrows():
        ev, mo = row["eve_pct"], row["mor_pct"]
        j = np.random.uniform(-0.05, 0.05)
        lc = cfg.UP_COLOR if mo >= ev else cfg.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=cfg.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=cfg.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
    y_top = max(max(eve_vals), max(mor_vals)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ==========================================================================
    # EXECUTION TOGGLES
    # ==========================================================================
    FORCE_PREPROCESSING               = False
    PLOT_INDIVIDUAL_TRIALS            = False
    PLOT_TRIAL_SCORE_SCATTER          = True
    PLOT_SUBJECT_AVERAGE_TIMECOURSE   = True
    PLOT_SUBJECT_MULTICHANNEL         = True   # per-subject all-channel grid
    PLOT_GROUP_AVERAGE_TIMECOURSE     = True
    PLOT_MULTICHANNEL_GROUP_TIMECOURSE = True  # all-channel group grid (key RESP plot)
    PLOT_GROUP_FOUR_CONDITIONS        = True
    PLOT_GROUP_RATIO                  = True
    PLOT_GROUP_SESSION_COMPARISON     = True
    PLOT_GROUP_EVE_MOR_RATIO          = True
    PLOT_GROUP_OVERALL_TIMECOURSE     = True
    PLOT_GROUP_OVERALL_EVE_VS_MOR     = True
    PLOT_GROUP_OVERALL_RATIO          = True
    PLOT_STAI_CORRELATIONS            = True
    PLOT_SUBJECTIVE_NEG_PERCENTAGE    = True
    # ==========================================================================

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(cfg.OUTPUT_DIR, "resp_signals_cache.pkl")

    # ── Load or build cache ───────────────────────────────────────────────────
    subj_data_dict: dict = {}
    cache_loaded   = False
    cache_meta     = {}

    if os.path.exists(cache_path) and not FORCE_PREPROCESSING:
        print("Loading respiration signals cache ...")
        try:
            with open(cache_path, "rb") as f:
                stored = pickle.load(f)
            cache_meta     = stored.get("meta", {})
            subj_data_dict = stored.get("data", {})
            cache_loaded   = True
        except Exception as e:
            print(f"  [!] Failed to load cache: {e}. Will rebuild.")

    # Check whether cache was built with compatible WIDE window and channels
    meta_ok = (
        cache_meta.get("wide_tmin") == cfg.WIDE_TMIN
        and cache_meta.get("wide_tmax") == cfg.WIDE_TMAX
        and set(cache_meta.get("channels", [])) == set(cfg.RESP_CHANNELS)
    )
    if cache_loaded and not meta_ok:
        print("  Cache parameters changed (WIDE_TMIN/TMAX or RESP_CHANNELS). Rebuilding.")
        subj_data_dict = {}
        cache_loaded   = False

    subject_folders = sorted([
        d for d in os.listdir(cfg.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(cfg.RAW_DATA_DIR, d))
    ])
    if cfg.SUBJECT_FILTER:
        subject_folders = [s for s in subject_folders if s in cfg.SUBJECT_FILTER]

    cache_changed = False
    for subj in subject_folders:
        subj_path  = os.path.join(cfg.RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        if not os.path.isdir(eeg_folder):
            continue

        startle_folder = find_startle_output_folder(subj_path)
        if not startle_folder:
            continue

        if not FORCE_PREPROCESSING and subj in subj_data_dict:
            # Validate cached channels match current config
            first_trial = next(
                (t for s in subj_data_dict[subj].values() for t in s), None
            )
            if (first_trial is not None
                    and set(cfg.RESP_CHANNELS).issubset(set(first_trial.get("channels", []))
                                                        | {ch for ch in cfg.RESP_CHANNELS
                                                           if ch not in first_trial["epoch_raw_wide"]})):
                continue  # cache still valid for this subject

        print(f"\n--- {subj} ---")
        subj_data_dict[subj] = {}
        cache_changed = True

        for sess_key, sess_info in cfg.SESSION_MAP.items():
            mff_path = find_mff_file(eeg_folder, sess_key)
            if not mff_path:
                print(f"  [!] No _{sess_key}_ MFF — skipping")
                continue
            csv_path = find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            if not csv_path:
                print(f"  [!] No {sess_info['csv_suffix']}.csv — skipping")
                continue

            print(f"  [{sess_info['label']}]")
            ratings_df = load_and_classify_ratings(csv_path)
            try:
                trials = process_session(mff_path, ratings_df)
            except Exception as e:
                print(f"    [!] Error: {e}")
                continue
            if trials:
                subj_data_dict[subj][sess_key] = trials

    if cache_changed or not cache_loaded:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "meta": {
                    "wide_tmin": cfg.WIDE_TMIN,
                    "wide_tmax": cfg.WIDE_TMAX,
                    "channels":  cfg.RESP_CHANNELS,
                },
                "data": subj_data_dict,
            }, f)
        print("\nCache updated.")

    # ── Always re-apply analysis parameters ──────────────────────────────────
    print("Applying analysis parameters (filter, baseline, rejection, scoring) ...")
    apply_analysis_params(subj_data_dict)

    # ── Rejection summary ─────────────────────────────────────────────────────
    print("\n== Rejection Summary ===========================================")
    for subj, sessions in sorted(subj_data_dict.items()):
        total_all = total_rej = 0
        for trials in sessions.values():
            total_all += len(trials)
            total_rej += sum(1 for t in trials if t["rejected"])
        pct = 100 * total_rej / max(total_all, 1)
        print(f"  {subj}: {total_all} trials | {total_rej} rejected ({pct:.0f}%)")
    print()

    # ── Plotting ──────────────────────────────────────────────────────────────
    if PLOT_INDIVIDUAL_TRIALS:
        print("Generating individual trial plots ...")
        plot_individual_trials(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_TRIAL_SCORE_SCATTER:
        print("\nGenerating trial-score scatter plots ...")
        plot_trial_scores_per_subject(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_SUBJECT_AVERAGE_TIMECOURSE:
        print("\nGenerating per-subject average timecourse plots ...")
        plot_subject_average_timecourse(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_SUBJECT_MULTICHANNEL:
        print("\nGenerating per-subject multi-channel timecourse plots ...")
        plot_subject_multichannel_timecourse(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_AVERAGE_TIMECOURSE:
        print("\nGenerating group average timecourse ...")
        plot_group_average_timecourse(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_MULTICHANNEL_GROUP_TIMECOURSE:
        print("\nGenerating multi-channel group timecourse ...")
        plot_multichannel_group_timecourse(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_FOUR_CONDITIONS:
        print("\nGenerating group four-condition boxplot ...")
        plot_group_boxplot_four_conditions(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_RATIO:
        print("\nGenerating Neg/Neu ratio boxplot ...")
        plot_group_ratio_boxplot(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_SESSION_COMPARISON:
        print("\nGenerating session-comparison boxplot ...")
        plot_group_session_comparison(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_EVE_MOR_RATIO:
        print("\nGenerating Eve/Mor ratio boxplot ...")
        plot_group_eve_mor_ratio(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_TIMECOURSE:
        print("\nGenerating overall timecourse (Eve vs Mor) ...")
        plot_group_overall_timecourse(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_EVE_VS_MOR:
        print("\nGenerating overall Eve vs Mor boxplot ...")
        plot_group_overall_eve_vs_mor(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_RATIO:
        print("\nGenerating overall Eve/Mor ratio ...")
        plot_group_overall_ratio(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_STAI_CORRELATIONS:
        print("\nGenerating STAI-T correlations ...")
        plot_stai_correlations(subj_data_dict, cfg.OUTPUT_DIR)

    if PLOT_SUBJECTIVE_NEG_PERCENTAGE:
        print("\nGenerating subjective negative% boxplot ...")
        plot_subjective_negative_percentage(cfg.RAW_DATA_DIR, cfg.OUTPUT_DIR)

    print("\nDone.")
    subprocess.run(["open", cfg.OUTPUT_DIR])


if __name__ == "__main__":
    main()
