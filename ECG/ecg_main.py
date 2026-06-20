"""
ECG Main Pipeline
=================
Orchestration: cache management, plotting, and group-level analysis.
Mirrors emg_raw_potentiation.py structure with ECG-specific adaptations.
"""

import os
import pickle
import subprocess
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal, stats

from ECG import ecg_config as config
from ECG import ecg_processor as processor

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sig_label(pval):
    return "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))


def _draw_stat_bracket(ax, x1, x2, y_top, pval, test_label, fontsize=10):
    """Draw a horizontal bracket at y_top with significance text above it."""
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2],
            [y_top, y_top + h, y_top + h, y_top],
            lw=1.2, color="black")
    sig = _sig_label(pval)
    ax.text((x1 + x2) / 2, y_top + h * 1.5,
            f"{sig}  {test_label} p={pval:.3f}",
            ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


# ── Stage 0: raw inspection ───────────────────────────────────────────────────

def plot_raw_inspection(mff_path, subj, sess_key, channel, output_dir):
    """
    Stage 0: MNE interactive raw viewer for ECG inspection.
    Uses raw.plot() which handles scrolling, scaling, and event markers natively.
    Also saves a static HP-filtered PNG around the first startle.
    Close the interactive window to continue pipeline execution.
    """
    import mne
    from scipy.signal import butter, filtfilt

    print(f"    Loading {os.path.basename(mff_path)} (may take ~30 s) ...")
    raw   = mne.io.read_raw_egi(mff_path, preload=True, verbose=False,
                                 events_as_annotations=False)
    sfreq = raw.info["sfreq"]

    if channel not in raw.ch_names:
        print(f"    [!] Channel '{channel}' not in file. Available: {raw.ch_names[:10]}")
        return

    events_df = processor.get_events_from_eeg(raw)

    # Build MNE events array [sample, 0, event_id] for colored overlays
    mne_events_list = []
    for _, row in events_df.iterrows():
        code_str = row["Channel"].replace("D", "")
        try:
            mne_events_list.append([int(row["Sample"]), 0, int(code_str)])
        except ValueError:
            pass
    mne_events = (np.array(sorted(mne_events_list), dtype=int)
                  if mne_events_list else None)

    # Build color dict for every event code present — raw.plot() errors on missing keys
    named = {
        config.TRIGGER_STARTLE:       "blue",
        config.TRIGGER_SESSION_START: "green",
        config.TRIGGER_SESSION_END:   "red",
    }
    all_ids = (set(mne_events[:, 2].tolist()) if mne_events is not None else set())
    event_color = {eid: named.get(eid, "lightgrey") for eid in all_ids}

    d110_samps = events_df[
        events_df["Channel"] == f"D{config.TRIGGER_STARTLE}"
    ]["Sample"].values
    start_time = max(0.0, (d110_samps[0] / sfreq) - 5) if len(d110_samps) > 0 else 0.0

    sess_label = config.SESSION_MAP[sess_key]["label"]
    raw_ecg = raw.copy().pick([channel])

    print(f"    Opening interactive viewer — use ← → to scroll, scroll wheel to zoom, close to continue")
    print(f"    [green=session start | red=session end | blue=startle probe]")

    # MNE's raw.plot() handles proper downsampling, scrollable timeline, event markers
    raw_ecg.plot(
        events=mne_events,
        event_color=event_color,
        duration=30.0,
        start=start_time,
        scalings="auto",
        title=f"{subj}  |  {sess_label}  |  {channel}   "
              f"[green=start  blue=startle  red=end]",
        block=True,
        show=True,
    )

    # Static PNG: HP-filtered signal around first startle (cardiac morphology visible)
    ch_data = raw_ecg.get_data()[0]
    times_s = np.arange(len(ch_data)) / sfreq
    nyq = sfreq / 2.0
    b, a = butter(config.HP_FILTER_ORDER, config.HP_CUTOFF_HZ / nyq, btype="high")
    filt_uv = filtfilt(b, a, ch_data) * 1e6

    if len(d110_samps) > 0:
        first_t = d110_samps[0] / sfreq
        t_lo = max(0.0, first_t - 10.0)
        t_hi = min(times_s[-1], first_t + 10.0)
    else:
        t_lo, t_hi = 0.0, min(times_s[-1], 20.0)
        first_t = None

    mask = (times_s >= t_lo) & (times_s <= t_hi)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(times_s[mask], filt_uv[mask], color="#2C3E50", linewidth=0.8)
    if first_t is not None:
        ax.axvline(first_t, color="blue", linewidth=1.2, label=f"D{config.TRIGGER_STARTLE} startle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HP-filtered ECG (µV)")
    ax.set_title(f"HP-filtered ECG — {subj} | {sess_label} | ±10 s around first startle")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    insp_dir = os.path.join(output_dir, "raw_inspection")
    os.makedirs(insp_dir, exist_ok=True)
    fname = os.path.join(insp_dir, f"{subj}_{sess_key}_raw_ecg.png")
    fig.savefig(fname, dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved raw inspection PNG: {fname}")


# ── Individual trial plots ────────────────────────────────────────────────────

def plot_individual_trials(subj_data_dict, output_dir):
    """
    For each subject/session create one PNG per trial showing wide-window
    filtered ECG with R-peak markers. Rejected = grey. Neg = red, Neu = blue.
    """
    base_dir = os.path.join(output_dir, "individual_trials")

    for subj, sessions in sorted(subj_data_dict.items()):
        n_rejected_total = 0
        n_total = 0

        for sess_key in ["mor", "eve"]:
            trials = sessions.get(sess_key)
            if not trials:
                continue

            sess_dir = os.path.join(base_dir, subj, sess_key)
            os.makedirs(sess_dir, exist_ok=True)

            for i, t in enumerate(trials):
                n_total += 1
                rejected = t["rejected"]
                if rejected:
                    n_rejected_total += 1
                    color = config.REJ_COLOR
                    cond_name = "REJECTED"
                else:
                    if t["label"] == 1:
                        color = config.NEG_COLOR
                        cond_name = "Negative"
                    else:
                        color = config.NEU_COLOR
                        cond_name = "Neutral"

                fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

                axes[0].plot(t["times_snippet"] * 1000, t["ecg_raw_snippet"] * 1e6,
                             color=color, linewidth=1.2, alpha=0.85)
                axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
                axes[0].axvspan(config.BASELINE_TMIN * 1000, config.BASELINE_TMAX * 1000,
                                color="gold", alpha=0.15, label="Baseline window")
                axes[0].set_ylabel("Raw (µV)")
                axes[0].set_title("Raw Signal (before filtering)")
                axes[0].legend(fontsize=8, loc="upper right")
                axes[0].grid(True, linestyle="--", alpha=0.4)

                axes[1].plot(t["times_snippet"] * 1000, t["ecg_snippet"] * 1e6,
                             color=color, linewidth=1.5)
                axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
                axes[1].axvspan(config.BASELINE_TMIN * 1000, config.BASELINE_TMAX * 1000,
                                color="gold", alpha=0.15, label="Baseline window")
                axes[1].axvspan(config.SCORE_TMIN * 1000, config.SCORE_TMAX * 1000,
                                color="green", alpha=0.10,
                                label=f"Score window ({config.SCORE_TMIN*1000:.0f}–{config.SCORE_TMAX*1000:.0f} ms)")
                if not rejected and len(t["r_peaks_epoch"]) > 0:
                    rp_t = t["r_peaks_epoch"] / t["sfreq"] * 1000 + config.WIDE_TMIN * 1000
                    rp_v = t["ecg_snippet"][t["r_peaks_epoch"]] * 1e6
                    axes[1].scatter(rp_t, rp_v, color="red", s=18, zorder=5, label="R-peaks")
                axes[1].set_ylabel("Processed (µV)")
                axes[1].set_xlabel("Time (ms)")
                axes[1].set_title(
                    f"Processed (HP > {config.HP_CUTOFF_HZ} Hz → LP < {config.LP_CUTOFF_HZ} Hz)")
                axes[1].legend(fontsize=8, loc="upper right")
                axes[1].grid(True, linestyle="--", alpha=0.4)

                score_str = f"{t['hr_score']:.2f} bpm" if not rejected else "REJECTED"
                suptitle = (f"{subj} | {sess_key.upper()} | Trial {i+1} | {cond_name}\n"
                            f"Image: {t['image_detail']} | Val: {t['valence']} | "
                            f"Ar: {t['arousal']} | Score: {score_str}")
                fig.suptitle(suptitle, fontsize=11, fontweight="bold")
                plt.tight_layout()

                fname = os.path.join(sess_dir,
                                     f"trial_{i+1:03d}_{cond_name[:3].upper()}.png")
                fig.savefig(fname, dpi=130, bbox_inches="tight")
                plt.close(fig)

        n_accepted = n_total - n_rejected_total
        print(f"  {subj}: {n_total} trials, "
              f"{n_rejected_total} rejected ({100*n_rejected_total/max(n_total,1):.0f}%), "
              f"{n_accepted} accepted")


# ── Trial-number scatter per subject ──────────────────────────────────────────

def plot_trial_scores_per_subject(subj_data_dict, output_dir):
    """One figure per subject: trial number vs HR score, coloured by Neg/Neu."""
    plot_dir = os.path.join(output_dir, "trial_score_scatter")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        fig.suptitle(f"{subj} – Trial Score by Trial Number", fontsize=13, fontweight="bold")

        for ax, sess_key in zip(axes, ["mor", "eve"]):
            trials = sessions.get(sess_key)
            if not trials:
                ax.set_title(f"{sess_key.upper()} – no data")
                continue

            scores    = np.array([t["hr_score"] for t in trials])
            labels    = np.array([t["label"] for t in trials])
            rejected  = np.array([t["rejected"] for t in trials])
            trial_nums = np.arange(1, len(trials) + 1)

            for cond, color, name in [(1, config.NEG_COLOR, "Negative"),
                                      (2, config.NEU_COLOR, "Neutral")]:
                mask = (~rejected) & (labels == cond)
                n_trials = mask.sum()
                if n_trials > 0:
                    ax.scatter(trial_nums[mask], scores[mask], color=color,
                               s=60, edgecolors="white", linewidths=0.5, zorder=3,
                               label=f"{name} (n={n_trials})")

            n_rej = rejected.sum()
            if n_rej > 0:
                ax.scatter(trial_nums[rejected], np.zeros(n_rej),
                           color=config.REJ_COLOR, marker="x", s=80, zorder=2,
                           label=f"Rejected (n={n_rej})")

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"{config.SESSION_MAP[sess_key]['label']}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Trial number")
            if ax == axes[0]:
                ax.set_ylabel("HR Score (bpm – peak minus baseline)")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{subj}_trial_scores.png")
        fig.savefig(fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved trial-score scatter: {fname}")


# ── Group boxplot: 4 conditions ───────────────────────────────────────────────

def plot_group_boxplot_four_conditions(subj_data_dict, output_dir):
    """
    Boxplot: Evening-Neg | Evening-Neu | Morning-Neg | Morning-Neu.
    Connecting lines within subject per session. Wilcoxon brackets.
    """
    os.makedirs(output_dir, exist_ok=True)

    cond_keys = [
        ("eve", 1, "Eve-Neg", config.NEG_COLOR, "#E8A99A"),
        ("eve", 2, "Eve-Neu", config.NEU_COLOR, "#9AC4E8"),
        ("mor", 1, "Mor-Neg", "#922B21",        "#E8A99A"),
        ("mor", 2, "Mor-Neu", "#1A5276",        "#9AC4E8"),
    ]

    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sess_key, cond, *_ in cond_keys:
            trials = sessions.get(sess_key)
            if not trials:
                continue
            vals = [t["hr_score"] for t in trials
                    if not t["rejected"] and t["label"] == cond]
            if vals:
                subj_means[subj][(sess_key, cond)] = np.mean(vals)

    valid_subjs = [s for s in subj_means
                   if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    n_subj = len(valid_subjs)

    data_cols = [[subj_means[s][(sk, c)] for s in valid_subjs]
                 for sk, c, *_ in cond_keys]

    n_trials_per_cond = []
    for sess_key, cond, *_ in cond_keys:
        all_t = []
        for subj in valid_subjs:
            all_t += [t for t in subj_data_dict[subj].get(sess_key, [])
                      if not t["rejected"] and t["label"] == cond]
        n_trials_per_cond.append(len(all_t))

    fig, ax = plt.subplots(figsize=(10, 6))
    positions  = [1, 2, 3, 4]
    box_colors = [c[3] for c in cond_keys]
    labels_x   = [c[2] for c in cond_keys]

    bp = ax.boxplot(data_cols, positions=positions, widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.7); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for s in valid_subjs:
        j = np.random.uniform(-0.07, 0.07)
        v_en = subj_means[s].get(("eve", 1))
        v_eu = subj_means[s].get(("eve", 2))
        if v_en is not None and v_eu is not None:
            lc = config.UP_COLOR if v_eu >= v_en else config.DOWN_COLOR
            ax.plot([1+j, 2+j], [v_en, v_eu], color=lc, alpha=0.45, linewidth=1.2)
            ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(2+j, v_eu, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        v_mn = subj_means[s].get(("mor", 1))
        v_mu = subj_means[s].get(("mor", 2))
        if v_mn is not None and v_mu is not None:
            lc = config.UP_COLOR if v_mu >= v_mn else config.DOWN_COLOR
            ax.plot([3+j, 4+j], [v_mn, v_mu], color=lc, alpha=0.45, linewidth=1.2)
            ax.scatter(3+j, v_mn, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(4+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    eve_neg, eve_neu, mor_neg, mor_neu = data_cols
    try:
        _, p_eve = stats.wilcoxon(eve_neg, eve_neu); tl_eve = "Wilcoxon"
    except Exception:
        _, p_eve = stats.ttest_rel(eve_neg, eve_neu); tl_eve = "Paired t"
    try:
        _, p_mor = stats.wilcoxon(mor_neg, mor_neu); tl_mor = "Wilcoxon"
    except Exception:
        _, p_mor = stats.ttest_rel(mor_neg, mor_neu); tl_mor = "Paired t"

    ax.set_ylim(bottom=ax.get_ylim()[0])
    y_top_eve = max(max(eve_neg), max(eve_neu)) * 1.10
    y_top_mor = max(max(mor_neg), max(mor_neu)) * 1.10
    ax.set_ylim(top=max(y_top_eve, y_top_mor) * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top_eve, p_eve, tl_eve)
    _draw_stat_bracket(ax, 3, 4, y_top_mor, p_mor, tl_mor)

    tick_labels = [f"{l}\n(n trials={nt})" for l, nt in zip(labels_x, n_trials_per_cond)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("HR Change Score (bpm)", fontsize=11)
    ax.set_title(f"Group HR Scores by Condition (N={n_subj} subjects)\n"
                 f"Lines connect Neg↔Neu within each session per subject",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.axvline(2.5, color="lightgray", linestyle="--", linewidth=1)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_four_conditions_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved four-condition group boxplot: {fname}")


# ── Group ratio boxplot (Neg/Neu per session) ─────────────────────────────────

def plot_group_ratio_boxplot(subj_data_dict, output_dir):
    """Neg/Neu HR score ratio per session (Evening | Morning). Wilcoxon bracket."""
    os.makedirs(output_dir, exist_ok=True)

    eve_ratios, mor_ratios, valid_subjs = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        ratios = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key)
            if not trials:
                continue
            neg_scores = [t["hr_score"] for t in trials
                          if not t["rejected"] and t["label"] == 1]
            neu_scores = [t["hr_score"] for t in trials
                          if not t["rejected"] and t["label"] == 2]
            if neg_scores and neu_scores and np.mean(neu_scores) != 0:
                ratios[sess_key] = np.mean(neg_scores) / np.mean(neu_scores)

        if "eve" in ratios and "mor" in ratios:
            eve_ratios.append(ratios["eve"])
            mor_ratios.append(ratios["mor"])
            valid_subjs.append(subj)

    n_subj = len(valid_subjs)
    if n_subj < 2:
        print("  [!] Not enough subjects for ratio plot")
        return

    try:
        _, pval = stats.wilcoxon(eve_ratios, mor_ratios); test_label = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_ratios, mor_ratios); test_label = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_ratios, mor_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for ev, mo in zip(eve_ratios, mor_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_ratios), max(mor_ratios)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n_subj})", f"Morning\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Neg / Neu HR Score Ratio", fontsize=11)
    ax.set_title(f"Negative-to-Neutral Ratio: Evening vs Morning\n"
                 f"(N={n_subj} subjects, lines = individual subjects)",
                 fontsize=12, fontweight="bold")
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_neg_neu_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ratio boxplot: {fname}")


# ── Group average timecourse ──────────────────────────────────────────────────

def plot_group_average_timecourse(subj_data_dict, output_dir):
    """
    Grand-average ± SEM HR timecourse (Neg/Neu split) for each session.
    Baseline-subtracted: each trial's timecourse minus its baseline_hr.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        f"Group Average HR Timecourse (accepted trials, "
        f"{config.ANAL_TMIN*1000:.0f} to {config.ANAL_TMAX*1000:.0f} ms)",
        fontsize=13, fontweight="bold")

    for ax, sess_key in zip(axes, ["eve", "mor"]):
        sess_label = config.SESSION_MAP[sess_key]["label"]
        neg_epochs, neu_epochs = [], []
        n_subjs_neg, n_subjs_neu = 0, 0

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key)
            if not trials:
                continue
            subj_neg = [t["hr_timecourse"] - t["baseline_hr"]
                        for t in trials if not t["rejected"] and t["label"] == 1]
            subj_neu = [t["hr_timecourse"] - t["baseline_hr"]
                        for t in trials if not t["rejected"] and t["label"] == 2]
            if subj_neg:
                neg_epochs.append(np.mean(subj_neg, axis=0)); n_subjs_neg += 1
            if subj_neu:
                neu_epochs.append(np.mean(subj_neu, axis=0)); n_subjs_neu += 1

        times_ms = None
        for _, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key)
            if trials:
                for t in trials:
                    if t["times_hr"] is not None:
                        times_ms = t["times_hr"] * 1000
                        break
            if times_ms is not None:
                break

        if times_ms is None or (not neg_epochs and not neu_epochs):
            ax.set_title(f"{sess_label} – no data")
            continue

        for epochs, color, name, n_subjs in [
            (neg_epochs, config.NEG_COLOR, "Negative", n_subjs_neg),
            (neu_epochs, config.NEU_COLOR, "Neutral",  n_subjs_neu),
        ]:
            if not epochs:
                continue
            arr = np.array(epochs)
            avg = np.mean(arr, axis=0)
            sem = stats.sem(arr, axis=0, nan_policy="omit")
            n_trials_total = sum(
                len([t for t in sessions.get(sess_key, [])
                     if not t["rejected"] and t["label"] == (1 if name == "Negative" else 2)])
                for _, sessions in subj_data_dict.items()
            )
            ax.plot(times_ms, avg, color=color, linewidth=2.2,
                    label=f"{name} (n subj={n_subjs}, n trials={n_trials_total})")
            ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
        ax.axvspan(config.SCORE_TMIN * 1000, config.SCORE_TMAX * 1000,
                   color="green", alpha=0.08,
                   label=f"Score window ({config.SCORE_TMIN*1000:.0f}–{config.SCORE_TMAX*1000:.0f} ms)")
        ax.set_title(sess_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("HR Change (bpm)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(output_dir, "group_average_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group average timecourse: {fname}")


# ── Subject average timecourse ────────────────────────────────────────────────

def plot_subject_average_timecourse(subj_data_dict, output_dir):
    """Per-subject average ± SEM HR timecourse (Neg/Neu split) for each session."""
    plot_dir = os.path.join(output_dir, "subject_average_timecourses")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(
            f"{subj} – Average HR Timecourse (accepted trials, "
            f"{config.ANAL_TMIN*1000:.0f} to {config.ANAL_TMAX*1000:.0f} ms)",
            fontsize=13, fontweight="bold")

        has_any_data = False

        for ax, sess_key in zip(axes, ["eve", "mor"]):
            sess_label = config.SESSION_MAP[sess_key]["label"]
            trials = sessions.get(sess_key, [])

            if not trials:
                ax.set_title(f"{sess_label} – no data")
                continue

            neg_epochs = [t["hr_timecourse"] - t["baseline_hr"]
                          for t in trials if not t["rejected"] and t["label"] == 1]
            neu_epochs = [t["hr_timecourse"] - t["baseline_hr"]
                          for t in trials if not t["rejected"] and t["label"] == 2]

            times_ms = None
            for t in trials:
                if t["times_hr"] is not None:
                    times_ms = t["times_hr"] * 1000
                    break

            if times_ms is None:
                ax.set_title(f"{sess_label} – no data")
                continue

            for epochs, color, name in [
                (neg_epochs, config.NEG_COLOR, "Negative"),
                (neu_epochs, config.NEU_COLOR, "Neutral"),
            ]:
                if not epochs:
                    continue
                has_any_data = True
                arr = np.array(epochs)
                avg = np.mean(arr, axis=0)
                sem = (stats.sem(arr, axis=0, nan_policy="omit")
                       if len(epochs) > 1 else np.zeros_like(avg))
                ax.plot(times_ms, avg, color=color, linewidth=2.2,
                        label=f"{name} (n trials={len(epochs)})")
                if len(epochs) > 1:
                    ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

            ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
            ax.axvspan(config.BASELINE_TMIN * 1000, config.BASELINE_TMAX * 1000,
                       color="gold", alpha=0.12, label="Baseline window")
            ax.axvspan(config.SCORE_TMIN * 1000, config.SCORE_TMAX * 1000,
                       color="green", alpha=0.08,
                       label=f"Score window ({config.SCORE_TMIN*1000:.0f}–{config.SCORE_TMAX*1000:.0f} ms)")
            ax.set_title(sess_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("HR Change (bpm, baseline-corrected)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        if has_any_data:
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_average_timecourse.png")
            fig.savefig(fname, dpi=160, bbox_inches="tight")
            print(f"  Saved subject average timecourse: {fname}")

        plt.close(fig)


# ── Subject PSD ───────────────────────────────────────────────────────────────

def plot_subject_psd(subj_data_dict, output_dir):
    """PSD of the raw unfiltered ECG: Evening (orange) and Morning (purple)."""
    plot_dir = os.path.join(output_dir, "subject_psd")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        has_any_data = False

        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            if not trials:
                continue
            first_t = trials[0]
            freqs = first_t.get("psd_freqs")
            psd   = first_t.get("psd_values")
            if freqs is None or psd is None:
                continue
            has_any_data = True
            color = config.EVE_COLOR if sess_key == "eve" else config.MOR_COLOR
            label = f"{config.SESSION_MAP[sess_key]['label']} ({first_t.get('channel', 'unknown')})"
            ax.semilogy(freqs, psd * 1e12, color=color, linewidth=1.5, label=label)

        if has_any_data:
            ax.set_xlabel("Frequency (Hz)", fontsize=11)
            ax.set_ylabel("Power Spectral Density (µV²/Hz)", fontsize=11)
            ax.set_title(f"{subj} – Power Spectral Density (Raw signal before filtering)",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            max_freq = max(freqs) if len(freqs) > 0 else 250
            ax.set_xlim(0, max_freq)
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_psd.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved subject PSD plot: {fname}")

        plt.close(fig)


# ── Session comparison boxplot ────────────────────────────────────────────────

def plot_group_session_comparison(subj_data_dict, output_dir):
    """
    Eve-Neg | Mor-Neg | Eve-Neu | Mor-Neu. Connecting lines per subject.
    Wilcoxon brackets for Neg pair and Neu pair.
    """
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
        for sess_key, cond, *_ in cond_keys:
            trials = sessions.get(sess_key, [])
            vals = [t["hr_score"] for t in trials
                    if not t["rejected"] and t["label"] == cond]
            if vals:
                subj_means[subj][(sess_key, cond)] = np.mean(vals)

    valid_subjs = [s for s in subj_means
                   if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    n_subj = len(valid_subjs)

    data_cols = [[subj_means[s][(sk, c)] for s in valid_subjs]
                 for sk, c, *_ in cond_keys]

    n_trials_per_cond = []
    for sess_key, cond, *_ in cond_keys:
        tot = sum(1 for subj in valid_subjs
                  for t in subj_data_dict[subj].get(sess_key, [])
                  if not t["rejected"] and t["label"] == cond)
        n_trials_per_cond.append(tot)

    fig, ax = plt.subplots(figsize=(10, 6))
    positions  = [1, 2, 3.6, 4.6]
    box_colors = [c[3] for c in cond_keys]
    bp = ax.boxplot(data_cols, positions=positions, widths=0.42,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.65); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for s in valid_subjs:
        j = np.random.uniform(-0.07, 0.07)
        v_en = subj_means[s][("eve", 1)]; v_mn = subj_means[s][("mor", 1)]
        lc = config.UP_COLOR if v_mn >= v_en else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [v_en, v_mn], color=lc, alpha=0.55, linewidth=1.4)
        ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(2+j, v_mn, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

        v_eu = subj_means[s][("eve", 2)]; v_mu = subj_means[s][("mor", 2)]
        lc = config.UP_COLOR if v_mu >= v_eu else config.DOWN_COLOR
        ax.plot([3.6+j, 4.6+j], [v_eu, v_mu], color=lc, alpha=0.55, linewidth=1.4)
        ax.scatter(3.6+j, v_eu, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(4.6+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    neg_eve, neg_mor, neu_eve, neu_mor = data_cols
    try:
        _, p_neg = stats.wilcoxon(neg_eve, neg_mor); tl_neg = "Wilcoxon"
    except Exception:
        _, p_neg = stats.ttest_rel(neg_eve, neg_mor); tl_neg = "Paired t"
    try:
        _, p_neu = stats.wilcoxon(neu_eve, neu_mor); tl_neu = "Wilcoxon"
    except Exception:
        _, p_neu = stats.ttest_rel(neu_eve, neu_mor); tl_neu = "Paired t"

    ax.set_ylim(bottom=ax.get_ylim()[0])
    y_top_neg = max(max(neg_eve), max(neg_mor)) * 1.10
    y_top_neu = max(max(neu_eve), max(neu_mor)) * 1.10
    ax.set_ylim(top=max(y_top_neg, y_top_neu) * 1.20)
    _draw_stat_bracket(ax, 1,   2,   y_top_neg, p_neg, tl_neg)
    _draw_stat_bracket(ax, 3.6, 4.6, y_top_neu, p_neu, tl_neu)

    tick_labels = [f"{lbl}\n(n={nt})"
                   for (_, _, lbl, _), nt in zip(cond_keys, n_trials_per_cond)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("HR Change Score (bpm – peak minus baseline)", fontsize=11)
    ax.set_title(f"Evening vs Morning by Condition  (N={n_subj} subjects)\n"
                 f"Green line = score increased, Red line = score decreased",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.axvline(2.8, color="lightgray", linestyle="--", linewidth=1)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_session_comparison_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved session-comparison boxplot: {fname}")


# ── Eve/Mor ratio per condition ───────────────────────────────────────────────

def plot_group_eve_mor_ratio(subj_data_dict, output_dir):
    """Neg ratio (Eve/Mor) vs Neu ratio (Eve/Mor). Wilcoxon bracket."""
    os.makedirs(output_dir, exist_ok=True)

    neg_ratios, neu_ratios, valid_subjs = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        r = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            for cond in [1, 2]:
                vals = [t["hr_score"] for t in trials
                        if not t["rejected"] and t["label"] == cond]
                if vals:
                    r[(sess_key, cond)] = np.mean(vals)

        if all(k in r for k in [("eve",1),("mor",1),("eve",2),("mor",2)]):
            if r[("mor", 1)] != 0 and r[("mor", 2)] != 0:
                neg_ratios.append(r[("eve", 1)] / r[("mor", 1)])
                neu_ratios.append(r[("eve", 2)] / r[("mor", 2)])
                valid_subjs.append(subj)

    n_subj = len(valid_subjs)
    if n_subj < 2:
        print("  [!] Not enough subjects for Eve/Mor ratio plot")
        return

    try:
        _, pval = stats.wilcoxon(neg_ratios, neu_ratios); test_label = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(neg_ratios, neu_ratios); test_label = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([neg_ratios, neu_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.NEG_COLOR, config.NEU_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for neg_r, neu_r in zip(neg_ratios, neu_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if neu_r >= neg_r else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [neg_r, neu_r], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, neg_r, color=config.NEG_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, neu_r, color=config.NEU_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(neg_ratios), max(neu_ratios)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Negative\n(n={n_subj})", f"Neutral\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Evening / Morning HR Score Ratio", fontsize=11)
    ax.set_title(f"Evening-to-Morning Ratio: Neg vs Neu  (N={n_subj} subjects)\n"
                 f"Green line = ratio increased Neg→Neu, Red = decreased",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_eve_mor_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Eve/Mor ratio boxplot: {fname}")


# ── Overall Eve vs Mor boxplot ────────────────────────────────────────────────

def plot_group_overall_eve_vs_mor(subj_data_dict, output_dir):
    """All accepted trials (no neg/neu split): Evening | Morning. Wilcoxon bracket."""
    os.makedirs(output_dir, exist_ok=True)

    eve_means, mor_means, valid_subjs = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sess_key in ["eve", "mor"]:
            scores = [t["hr_score"] for t in sessions.get(sess_key, [])
                      if not t["rejected"]]
            if scores:
                vals[sess_key] = np.mean(scores)
        if "eve" in vals and "mor" in vals:
            eve_means.append(vals["eve"]); mor_means.append(vals["mor"])
            valid_subjs.append(subj)

    n_subj = len(valid_subjs)
    if n_subj < 2:
        print("  [!] Not enough subjects for overall Eve vs Mor plot")
        return

    try:
        _, pval = stats.wilcoxon(eve_means, mor_means); test_label = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_means, mor_means); test_label = "Paired t"

    n_trials_eve = sum(
        sum(1 for t in subj_data_dict[s].get("eve", []) if not t["rejected"])
        for s in valid_subjs)
    n_trials_mor = sum(
        sum(1 for t in subj_data_dict[s].get("mor", []) if not t["rejected"])
        for s in valid_subjs)

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_means, mor_means], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for ev, mo in zip(eve_means, mor_means):
        j = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_means), max(mor_means)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Evening\n(n subj={n_subj}, n trials={n_trials_eve})",
         f"Morning\n(n subj={n_subj}, n trials={n_trials_mor})"], fontsize=10)
    ax.set_ylabel("HR Change Score (bpm)", fontsize=11)
    ax.set_title(f"Overall HR Score: Evening vs Morning\n"
                 f"(N={n_subj} subjects, all accepted trials, green=↑ red=↓)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_eve_vs_mor.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall Eve vs Mor boxplot: {fname}")


# ── Overall Eve/Mor ratio ─────────────────────────────────────────────────────

def plot_group_overall_ratio(subj_data_dict, output_dir):
    """One-sample test: is the overall Eve/Mor HR score ratio different from 1?"""
    os.makedirs(output_dir, exist_ok=True)

    ratios, valid_subjs = [], []
    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sess_key in ["eve", "mor"]:
            scores = [t["hr_score"] for t in sessions.get(sess_key, [])
                      if not t["rejected"]]
            if scores:
                vals[sess_key] = np.mean(scores)
        if "eve" in vals and "mor" in vals and vals["mor"] != 0:
            ratios.append(vals["eve"] / vals["mor"])
            valid_subjs.append(subj)

    n_subj = len(valid_subjs)
    if n_subj < 2:
        print("  [!] Not enough subjects for overall ratio plot")
        return

    try:
        _, pval = stats.wilcoxon([r - 1 for r in ratios]); test_label = "Wilcoxon (vs 1)"
    except Exception:
        _, pval = stats.ttest_1samp(ratios, 1); test_label = "One-sample t (vs 1)"

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
    ax.text(1, y_top * 1.05,
            f"{sig}\n{test_label}\np={pval:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks([1])
    ax.set_xticklabels([f"Eve / Mor\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Evening / Morning HR Score Ratio", fontsize=11)
    ax.set_title("Overall Eve/Mor Ratio (all accepted trials)\n"
                 "Tested against ratio=1 (ratio > 1 = Eve > Mor)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_ratio.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall ratio plot: {fname}")


# ── Overall average timecourse (Eve vs Mor) ───────────────────────────────────

def plot_group_overall_timecourse(subj_data_dict, output_dir):
    """Evening vs Morning grand-average timecourse across ALL accepted trials."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sess_styles = [
        ("eve", "Evening", config.EVE_COLOR),
        ("mor", "Morning", config.MOR_COLOR),
    ]

    times_ms = None
    for sess_key, sess_label, color in sess_styles:
        subj_avgs = []
        n_subjs = n_trials_total = 0

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            accepted = [t for t in trials if not t["rejected"]]
            if not accepted:
                continue
            epochs = [t["hr_timecourse"] - t["baseline_hr"] for t in accepted]
            subj_avgs.append(np.mean(epochs, axis=0))
            n_subjs += 1
            n_trials_total += len(accepted)
            if times_ms is None:
                for t in accepted:
                    if t["times_hr"] is not None:
                        times_ms = t["times_hr"] * 1000
                        break

        if not subj_avgs or times_ms is None:
            continue

        arr = np.array(subj_avgs)
        avg = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0, nan_policy="omit")
        ax.plot(times_ms, avg, color=color, linewidth=2.5,
                label=f"{sess_label} (n subj={n_subjs}, n trials={n_trials_total})")
        ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
    ax.axvspan(config.SCORE_TMIN * 1000, config.SCORE_TMAX * 1000,
               color="green", alpha=0.08,
               label=f"Score window ({config.SCORE_TMIN*1000:.0f}–{config.SCORE_TMAX*1000:.0f} ms)")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("HR Change (bpm, baseline-corrected)", fontsize=11)
    ax.set_title("Overall Average HR Timecourse: Evening vs Morning\n"
                 "(all accepted trials, ±SEM shaded)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall average timecourse: {fname}")


# ── STAI-T correlation grid ───────────────────────────────────────────────────

def plot_stai_correlations(subj_data_dict, output_dir, xlsx_path=None):
    """
    Load STAI-T from subjects spreadsheet and correlate with every HR and
    subjective measure. One scatter panel per measure, Pearson r overlaid.
    """
    if xlsx_path is None:
        xlsx_path = config.SUBJECTS_XLSX

    def safe_ratio(num, denom):
        if num is None or denom is None or pd.isna(num) or pd.isna(denom) or denom == 0:
            return np.nan
        return float(num) / float(denom)

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

        for sess_key, cond_label in [("eve", "neg"), ("eve", "neu"),
                                     ("mor", "neg"), ("mor", "neu")]:
            cond_val = 1 if cond_label == "neg" else 2
            trials = sessions.get(sess_key, [])
            vals = [t["hr_score"] for t in trials
                    if not t["rejected"] and t["label"] == cond_val]
            m[f"hr_{sess_key}_{cond_label}"] = np.mean(vals) if vals else np.nan

        m["hr_neg_neu_ratio"] = safe_mean([
            safe_ratio(m.get("hr_eve_neg"), m.get("hr_eve_neu")),
            safe_ratio(m.get("hr_mor_neg"), m.get("hr_mor_neu"))
        ])
        m["hr_eve_mor_ratio"] = safe_mean([
            safe_ratio(m.get("hr_eve_neg"), m.get("hr_mor_neg")),
            safe_ratio(m.get("hr_eve_neu"), m.get("hr_mor_neu"))
        ])
        m["hr_neu_eve_mor_ratio"] = safe_ratio(m.get("hr_eve_neu"), m.get("hr_mor_neu"))

        for sess_key, label in [("eve", "eve"), ("mor", "mor")]:
            trials = sessions.get(sess_key, [])
            neg_trials = [t for t in trials if not t["rejected"] and t["label"] == 1]
            m[f"val_{label}"]  = np.mean([t["valence"] for t in neg_trials]) if neg_trials else np.nan
            m[f"arsl_{label}"] = np.mean([t["arousal"] for t in neg_trials]) if neg_trials else np.nan

        m["val_ratio"]  = safe_ratio(m.get("val_eve"),  m.get("val_mor"))
        m["arsl_ratio"] = safe_ratio(m.get("arsl_eve"), m.get("arsl_mor"))

        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            if trials:
                neg_count = sum(1 for t in trials if t["label"] == 1)
                m[f"neg_pct_{sess_key}"] = 100.0 * neg_count / len(trials)
            else:
                m[f"neg_pct_{sess_key}"] = np.nan

        rows.append(m)

    if len(rows) < 3:
        print("  [!] Not enough subjects with STAI-T data for correlation plot")
        return

    df = pd.DataFrame(rows).set_index("subj")

    panels = [
        ("hr_eve_neg",           "HR Score (bpm)",           "Evening – Negative"),
        ("hr_eve_neu",           "HR Score (bpm)",           "Evening – Neutral"),
        ("hr_mor_neg",           "HR Score (bpm)",           "Morning – Negative"),
        ("hr_mor_neu",           "HR Score (bpm)",           "Morning – Neutral"),
        ("hr_neg_neu_ratio",     "Neg / Neu Ratio",          "Neg/Neu Ratio (avg sessions)"),
        ("hr_eve_mor_ratio",     "Eve / Mor Ratio",          "Eve/Mor Ratio (avg conditions)"),
        ("hr_neu_eve_mor_ratio", "Eve / Mor Neutral Ratio",  "Eve/Mor Neutral Ratio"),
        ("val_eve",              "Valence rating",           "Valence – Evening (Neg trials)"),
        ("val_mor",              "Valence rating",           "Valence – Morning (Neg trials)"),
        ("val_ratio",            "Valence Ratio Eve/Mor",    "Valence Ratio Eve/Mor"),
        ("arsl_eve",             "Arousal rating",           "Arousal – Evening (Neg trials)"),
        ("arsl_mor",             "Arousal rating",           "Arousal – Morning (Neg trials)"),
        ("arsl_ratio",           "Arousal Ratio Eve/Mor",    "Arousal Ratio Eve/Mor"),
        ("neg_pct_eve",          "% Neg trials",             "% Neg trials (Evening)"),
    ]

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axes = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(panels):
        ax = axes[idx]
        y_vals = df[col].values.astype(float)
        x_vals = df["stai"].values.astype(float)
        mask   = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x, y   = x_vals[mask], y_vals[mask]
        n      = mask.sum()

        if n < 3 or np.std(x) == 0 or np.std(y) == 0:
            ax.set_title(title + "\n(no variance / constant data)", fontsize=10)
            ax.axis("off")
            continue

        r, p   = stats.pearsonr(x, y)
        sig    = _sig_label(p)

        if "neg" in col or "neg" in title.lower():
            dot_color = config.NEG_COLOR
        elif "neu" in col or "neu" in title.lower():
            dot_color = config.NEU_COLOR
        else:
            dot_color = "#7F8C8D"

        ax.scatter(x, y, color=dot_color, edgecolors="white",
                   s=60, linewidths=0.5, zorder=3, alpha=0.85)

        m_coef, b_coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m_coef * x_line + b_coef,
                color="#2C3E50", linewidth=1.6, linestyle="--", zorder=2)

        for subj_id, xi, yi in zip(df.index[mask], x, y):
            ax.annotate(subj_id, (xi, yi), fontsize=6, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points", color="#555555")

        p_str    = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        stat_txt = f"r={r:+.2f}  {p_str}  {sig}\nn={n}"
        ax.text(0.04, 0.97, stat_txt, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#CCCCCC", alpha=0.85))

        ax.set_xlabel("STAI-T score", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("STAI-T Correlations with ECG HR & Subjective Ratings",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "stai_correlations.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved STAI-T correlation grid: {fname}")


# ── Subjective negative percentage ───────────────────────────────────────────

def plot_subjective_negative_percentage(raw_data_dir, output_dir):
    """
    % of subjective Negative trials (from CSV ratings alone) for each session.
    No EEG data used — purely CSV-based. Eve vs Mor. Wilcoxon bracket.
    """
    subject_folders = sorted([
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ])

    data = []
    for subj in subject_folders:
        subj_path      = os.path.join(raw_data_dir, subj)
        startle_folder = processor.find_startle_output_folder(subj_path)
        if startle_folder is None:
            continue

        pcts = {}
        for sess_key, sess_info in config.SESSION_MAP.items():
            csv_path = processor.find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            if csv_path is None:
                continue
            try:
                df = processor.load_and_classify_ratings(csv_path)
                if len(df) > 0:
                    neg_count = sum(df["subjective_label"] == 1)
                    pcts[sess_key] = 100.0 * neg_count / len(df)
            except Exception as e:
                print(f"  [!] Error reading ratings for {subj} {sess_key}: {e}")

        if "eve" in pcts and "mor" in pcts:
            data.append({"subj": subj, "eve_pct": pcts["eve"], "mor_pct": pcts["mor"]})

    n_subj = len(data)
    if n_subj < 2:
        print("  [!] Not enough subjects with rating CSVs for percentage plot")
        return

    df_pcts  = pd.DataFrame(data)
    eve_vals = df_pcts["eve_pct"].values
    mor_vals = df_pcts["mor_pct"].values

    try:
        _, pval = stats.wilcoxon(eve_vals, mor_vals); test_label = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_vals, mor_vals); test_label = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_vals, mor_vals], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [config.EVE_COLOR, config.MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for _, row in df_pcts.iterrows():
        ev = row["eve_pct"]; mo = row["mor_pct"]
        j  = np.random.uniform(-0.05, 0.05)
        lc = config.UP_COLOR if mo >= ev else config.DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=config.EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=config.MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_vals), max(mor_vals)) * 1.05
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Evening\n(n={n_subj})", f"Morning\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Percent of subjective negative trials (%)", fontsize=11)
    ax.set_title(f"Subjective Negative Trials Percentage: Evening vs Morning\n"
                 f"(Based solely on CSV ratings, green = increase Eve→Mor)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "subjective_negative_percentage_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved subjective negative percentage boxplot: {fname}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    STAGE_0_RAW_INSPECTION           = False
    PLOT_INDIVIDUAL_TRIALS           = False
    PLOT_TRIAL_SCORE_SCATTER         = False
    PLOT_SUBJECT_AVERAGE_TIMECOURSE  = False
    PLOT_SUBJECT_PSD                 = False
    PLOT_GROUP_AVERAGE_TIMECOURSE    = True
    PLOT_GROUP_FOUR_CONDITIONS       = True
    PLOT_GROUP_RATIO                 = True
    PLOT_GROUP_SESSION_COMPARISON    = True
    PLOT_GROUP_EVE_MOR_RATIO         = True
    PLOT_GROUP_OVERALL_TIMECOURSE    = True
    PLOT_GROUP_OVERALL_EVE_VS_MOR    = True
    PLOT_GROUP_OVERALL_RATIO         = True
    PLOT_STAI_CORRELATIONS           = True
    PLOT_SUBJECTIVE_NEG_PERCENTAGE   = True

    os.makedirs(config.PLOT_OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(config.OUTPUT_DIR, "ecg_signals_cache.pkl")

    cache_loaded   = False
    subj_data_dict = {}
    if os.path.exists(cache_path) and not config.FORCE_RELOAD_ECG:
        print("Loading ECG signals cache ...")
        try:
            with open(cache_path, "rb") as f:
                subj_data_dict = pickle.load(f)
            cache_loaded = True
        except Exception as e:
            print(f"  [!] Failed to load cache: {e}. Will rebuild.")

    subject_folders = sorted([
        d for d in os.listdir(config.RAW_DATA_DIR)
        if os.path.isdir(os.path.join(config.RAW_DATA_DIR, d))
    ])

    cache_changed = False

    for subj in subject_folders:
        subj_path  = os.path.join(config.RAW_DATA_DIR, subj)
        eeg_folder = os.path.join(subj_path, "EEG")
        if not os.path.isdir(eeg_folder):
            eeg_folder = subj_path  # MFFs may be directly in subject folder

        startle_folder = processor.find_startle_output_folder(subj_path)
        if startle_folder is None:
            continue

        ch = config.SUBJECT_CHANNEL_OVERRIDES.get(subj, config.ECG_CHANNEL)

        use_cache = False
        if not config.FORCE_RELOAD_ECG and subj in subj_data_dict:
            cached_sessions = subj_data_dict[subj]
            if cached_sessions:
                first_trial = next(
                    (t for s in cached_sessions.values() for t in s), None)
                if (first_trial is not None
                        and first_trial.get("channel") == ch
                        and "psd_freqs" in first_trial):
                    use_cache = True

        if use_cache:
            continue

        print(f"\n--- {subj} (channel={ch}) ---")
        subj_data_dict[subj] = {}
        cache_changed = True

        for sess_key, sess_info in config.SESSION_MAP.items():
            mff_path = processor.find_mff_file(eeg_folder, sess_key)
            if mff_path is None:
                print(f"  [!] No _{sess_key}_ MFF - skipping")
                continue

            csv_path = processor.find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            if csv_path is None:
                print(f"  [!] No {sess_info['csv_suffix']}.csv - skipping")
                continue

            if STAGE_0_RAW_INSPECTION:
                print(f"  [Stage 0 raw inspection — {sess_info['label']}]")
                plot_raw_inspection(mff_path, subj, sess_key, ch, config.PLOT_OUTPUT_DIR)

            print(f"  [{sess_info['label']}]")
            ratings_df = processor.load_and_classify_ratings(csv_path)

            try:
                trials = processor.process_session(mff_path, ratings_df, ch, config)
            except Exception as e:
                print(f"    [!] Error: {e}")
                continue

            if trials:
                subj_data_dict[subj][sess_key] = trials

    if cache_changed or config.FORCE_RELOAD_ECG or not cache_loaded:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(subj_data_dict, f)
        print("\nECG signals cache updated.")

    print("Applying analysis parameters ...")
    processor.apply_analysis_params(subj_data_dict, config)

    print("\n== Rejection Summary =============================================")
    for subj, sessions in sorted(subj_data_dict.items()):
        total_rej = total_all = 0
        for sess_key, trials in sessions.items():
            total_all += len(trials)
            total_rej += sum(1 for t in trials if t["rejected"])
        pct = 100 * total_rej / max(total_all, 1)
        print(f"  {subj}: {total_all} trials | {total_rej} rejected ({pct:.0f}%)")
    print()

    if PLOT_INDIVIDUAL_TRIALS:
        print("Generating individual trial plots ...")
        plot_individual_trials(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_TRIAL_SCORE_SCATTER:
        print("Generating trial-score scatter plots ...")
        plot_trial_scores_per_subject(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_SUBJECT_AVERAGE_TIMECOURSE:
        print("Generating subject average timecourse plots ...")
        plot_subject_average_timecourse(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_SUBJECT_PSD:
        print("Generating subject PSD plots ...")
        plot_subject_psd(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_AVERAGE_TIMECOURSE:
        print("Generating group average timecourse ...")
        plot_group_average_timecourse(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_FOUR_CONDITIONS:
        print("Generating group four-condition boxplot ...")
        plot_group_boxplot_four_conditions(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_RATIO:
        print("Generating Neg/Neu ratio boxplot ...")
        plot_group_ratio_boxplot(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_SESSION_COMPARISON:
        print("Generating session-comparison boxplot ...")
        plot_group_session_comparison(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_EVE_MOR_RATIO:
        print("Generating Evening/Morning ratio boxplot ...")
        plot_group_eve_mor_ratio(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_STAI_CORRELATIONS:
        print("Generating STAI-T correlation grid ...")
        plot_stai_correlations(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_TIMECOURSE:
        print("Generating overall avg timecourse (Eve vs Mor) ...")
        plot_group_overall_timecourse(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_EVE_VS_MOR:
        print("Generating overall Eve vs Mor boxplot ...")
        plot_group_overall_eve_vs_mor(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_RATIO:
        print("Generating overall Eve/Mor ratio plot ...")
        plot_group_overall_ratio(subj_data_dict, config.PLOT_OUTPUT_DIR)

    if PLOT_SUBJECTIVE_NEG_PERCENTAGE:
        print("Generating subjective negative percentage boxplot ...")
        plot_subjective_negative_percentage(config.RAW_DATA_DIR, config.PLOT_OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
