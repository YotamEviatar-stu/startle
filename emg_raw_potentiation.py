"""
Raw Potentiation Approach – Startle EMG Analysis
=================================================
Pipeline per session:
  1. Load full session signal and high-pass filter (Butterworth 4th order, >28 Hz).
  2. For each startle trigger cut a wide window [-250 ms, +250 ms].
  3. Rectify and smooth the wide epoch (Butterworth 4th, <30 Hz).
  4. Compute the baseline = mean of [-50 ms, 0 ms].
     Reject trial if the signal in the 50 ms pre-stimulus window changes
     more than ±100 µV from the baseline or 3 sd.
  5. Trim accepted epochs to [-50 ms, +120 ms].
  6. Trial score = max peak in [20 ms, 100 ms] minus baseline mean.

Output:
  - Individual trial plots (raw + processed, rejected highlighted).
  - Trial-score-by-trial-number scatter per subject.
  - Group boxplots: Eve-Neg / Eve-Neu / Mor-Neg / Mor-Neu with connecting lines.
  - Group ratio boxplot: Neg/Neu ratio for Eve vs Mor with statistics.

Execution is toggled using boolean flags at the top of main().
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats, signal
import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Users/yotameviatar/Desktop/data"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/startle_output"

# Flag to determine trial type classification:
# True  = Subjective classification (arousal >= 7 or valence <= 3)
# False = Objective classification (from CSV "image_type" column)
USE_SUBJECTIVE_TRIAL_TYPE = False

# Save plots inside a subdirectory depending on subjective vs objective classification
if USE_SUBJECTIVE_TRIAL_TYPE:
    PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dynamic rejection")
else:
    PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "no_subjective")

EMG_CHANNEL = "E238"   # default channel for all subjects

# Per-subject channel overrides (leave empty to use EMG_CHANNEL for everyone).
# Example: { "NB03": "E241", "YR08": "E241" }
SUBJECT_CHANNEL_OVERRIDES: dict = {"NB03": "E241", "AS09": "E241", "MH20": "E241", "LO21": "E241"}

# Wide epoch for filtering / rejection assessment
WIDE_TMIN, WIDE_TMAX = -0.250, 0.250   # seconds

# Final analysis window  ← change freely; no re-preprocessing needed
ANAL_TMIN, ANAL_TMAX = -0.050, 0.120   # seconds

# Rejection threshold parameters  ← change freely; no re-preprocessing needed
ABSOLUTE_MAX_UV = 100e-6               # 100 µV absolute hard limit (e.g., electrode pop/disconnect)
Z_SCORE_THRESHOLD = 3.0                # dynamic limit: 3 standard deviations from session median

# Response scoring window (relative to time 0)  ← change freely
SCORE_WIN = (0.020, 0.100)             # 20–100 ms

# DIN trigger codes
TRIGGER_SESSION_START = 101
TRIGGER_SESSION_END   = 124
TRIGGER_STARTLE       = 110

# Session mapping
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# Colours
NEG_COLOR  = "#C0392B"   # deep red
NEU_COLOR  = "#2980B9"   # deep blue
REJ_COLOR  = "#95A5A6"   # grey for rejected
EVE_COLOR  = "#E67E22"   # orange shade for evening
MOR_COLOR  = "#8E44AD"   # purple shade for morning


# ── 1. File Discovery helpers ─────────────────────────────────────────────────

def find_startle_output_folder(subject_path):
    for name in os.listdir(subject_path):
        if name.lower() == "startle output" and os.path.isdir(os.path.join(subject_path, name)):
            return os.path.join(subject_path, name)
    return None


def find_mff_file(eeg_folder, keyword):
    for name in sorted(os.listdir(eeg_folder)):
        if f"_{keyword}_" in name.lower() and name.endswith(".mff"):
            return os.path.join(eeg_folder, name)
    return None


def find_csv_by_suffix(startle_folder, suffix):
    for name in os.listdir(startle_folder):
        if name.endswith(".csv") and "demo" not in name.lower():
            if re.search(rf"{suffix}\.csv$", name):
                return os.path.join(startle_folder, name)
    return None


# ── 2. DIN event extraction ───────────────────────────────────────────────────

def get_events_from_eeg(raw_eeg):
    din_names = [ch for ch in raw_eeg.ch_names if ch.startswith("D")]
    raw_din = raw_eeg.copy().pick(din_names)
    event_data = []
    for ch_idx, ch_name in enumerate(raw_din.ch_names):
        data = raw_din.get_data(picks=[ch_idx])[0]
        threshold = np.max(data) * 0.9
        for sample in np.where(data > threshold)[0]:
            event_data.append({"Channel": ch_name, "Sample": int(sample)})
    if not event_data:
        return pd.DataFrame(columns=["Channel", "Sample"])
    return pd.DataFrame(event_data).sort_values("Sample").reset_index(drop=True)


# ── 3. Ratings classification ─────────────────────────────────────────────────

def load_and_classify_ratings(csv_path: str):
    """Load a startle ratings CSV and classify Negative/Neutral trials."""
    df = pd.read_csv(csv_path)
    df = df[df["has_sound"] == True].reset_index(drop=True)

    def categorize(row):
        if row["arousalRating"] >= 7 or row["valenceRating"] <= 3:
            return 1  # Negative
        else:
            return 2  # Neutral

    df["subjective_label"] = df.apply(categorize, axis=1)
    return df


# ── 4. Core processing function ────────────────────────────────────────────────

def process_session(mff_path, ratings_df, channel=EMG_CHANNEL):
    """
    Load one session MFF, apply the raw-potentiation pipeline, and return
    per-trial raw signal data.

    IMPORTANT – Two-layer cache design
    ------------------------------------
    This function stores only the signals and metadata that require loading
    the MFF file.  Rejection, score, and analysis-window epoch are NOT stored
    here – they are derived by apply_analysis_params() on every run so they
    always reflect the current REJECT_THRESHOLD_UV / SCORE_WIN / ANAL_TMIN/MAX.

    Returns
    -------
    list of dicts, one per trial aligned to startle triggers:
      {
        "epoch_raw_wide"  : np.ndarray  (wide window, unfiltered),
        "epoch_proc_wide" : np.ndarray  (wide window, hp + rectify + lp),
        "times_wide"      : np.ndarray,
        "baseline_mean"   : float  (mean of [-50, 0) ms on proc signal),
        "label"           : int    (1=Neg, 2=Neu),
        "valence"         : float,
        "arousal"         : float,
        "image_detail"    : str,
        "sfreq"           : float,
        "channel"         : str,
        # ── filled in by apply_analysis_params(), never persisted ──
        # "rejected", "score", "epoch_proc_anal", "times_anal"
      }
    None on failure.
    """
    print(f"    Loading {os.path.basename(mff_path)} (channel={channel}) ...")
    raw = mne.io.read_raw_egi(mff_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # --- Crop to task segment ---
    events_df = get_events_from_eeg(raw)
    start_samps = events_df[events_df["Channel"] == f"D{TRIGGER_SESSION_START}"]["Sample"].values
    end_samps   = events_df[events_df["Channel"] == f"D{TRIGGER_SESSION_END}"]["Sample"].values
    if len(start_samps) == 0 or len(end_samps) == 0:
        print("    [!] Task boundaries not found - skipping")
        return None
    raw.crop(tmin=start_samps[0] / sfreq, tmax=end_samps[-1] / sfreq)

    # --- High-pass filter the full raw signal (Butterworth 4th, >28 Hz) ---
    ch_data_raw = raw.copy().pick([channel]).get_data()[0]

    # --- Compute Welch PSD of the raw unfiltered signal ---
    nperseg = int(min(len(ch_data_raw), 2048))
    psd_freqs, psd_values = signal.welch(ch_data_raw, fs=sfreq, nperseg=nperseg)

    nyquist = sfreq / 2.0
    b_hp, a_hp = signal.butter(4, 28 / nyquist, btype="high")
    sig_hp = signal.filtfilt(b_hp, a_hp, ch_data_raw)

    # --- Rectify then low-pass smooth (Butterworth 4th, <30 Hz) ---
    sig_rect = np.abs(sig_hp)
    b_lp, a_lp = signal.butter(4, 30 / nyquist, btype="low")
    sig_proc = signal.filtfilt(b_lp, a_lp, sig_rect)

    # --- Re-extract events after crop ---
    events_df = get_events_from_eeg(raw)
    startle_samps = events_df[events_df["Channel"] == f"D{TRIGGER_STARTLE}"]["Sample"].values

    n_use = min(len(startle_samps), len(ratings_df))
    startle_samps = startle_samps[:n_use]
    ratings_df    = ratings_df.iloc[:n_use].reset_index(drop=True)

    # --- Time axis for wide window ---
    wide_n     = int((WIDE_TMAX - WIDE_TMIN) * sfreq)
    times_wide = np.linspace(WIDE_TMIN, WIDE_TMAX, wide_n, endpoint=False)

    # Baseline window mask (fixed at -50…0 ms – not configurable)
    pre_mask = (times_wide >= -0.050) & (times_wide < 0.0)

    results = []
    for i, s_idx in enumerate(startle_samps):
        t0 = int(s_idx + WIDE_TMIN * sfreq)
        t1 = t0 + wide_n

        if t0 < 0 or t1 > len(sig_proc):
            continue

        ep_raw  = ch_data_raw[t0:t1]
        ep_proc = sig_proc[t0:t1]

        baseline_mean = float(np.mean(ep_proc[pre_mask]))

        row = ratings_df.iloc[i]
        img_col = "image" if "image" in row.index else (
                  "image_name" if "image_name" in row.index else "image_type")

        # Extract objective image_type / condition name
        img_type_val = ""
        for col in ["image_type", "file_name", "image", "image_name"]:
            if col in row.index and not pd.isna(row[col]):
                img_type_val = str(row[col])
                break

        results.append({
            # ── raw signals (MFF-derived, cached) ──
            "epoch_raw_wide":  ep_raw,
            "epoch_proc_wide": ep_proc,
            "times_wide":      times_wide,
            "baseline_mean":   baseline_mean,
            "sfreq":           sfreq,
            "channel":         channel,
            "psd_freqs":       psd_freqs,
            "psd_values":      psd_values,
            # ── metadata (cached) ──
            "subjective_label": int(row["subjective_label"]),
            "image_type":       img_type_val,
            "label":            int(row["subjective_label"]),
            "valence":          row["valenceRating"],
            "arousal":          row["arousalRating"],
            "image_detail":     str(row.get(img_col, "?")),
            # ── derived fields – populated by apply_analysis_params() ──
            "rejected":        None,
            "score":           None,
            "epoch_proc_anal": None,
            "times_anal":      None,
        })

    return results


# ── 5. Apply analysis parameters (fast, no MFF access needed) ───────────────

def apply_analysis_params(subj_data_dict):
    """
        (Re-)compute rejection, score, and trimmed analysis epoch for every trial
        in subj_data_dict using the current module-level parameters:

            ABSOLUTE_MAX_UV      – Hard microvolt limit for broken trials
            Z_SCORE_THRESHOLD    – Subject-specific deviation limit
            SCORE_WIN            – (t_start, t_end) in seconds for peak scoring
            ANAL_TMIN / ANAL_TMAX – analysis window edges
        """
    for subj, sessions in subj_data_dict.items():
        for sess_key, trials in sessions.items():
            if not trials:
                continue

            # --- Pass 1: Collect baseline metrics for the session ---
            baseline_peaks = []
            for t in trials:
                tw = t["times_wide"]
                ep = t["epoch_proc_wide"]
                pre_mask = (tw >= -0.050) & (tw < 0.0)

                # Save the maximum amplitude in the baseline for this trial
                trial_max = float(np.max(ep[pre_mask]))
                t["baseline_max"] = trial_max
                baseline_peaks.append(trial_max)

            # Compute session-specific statistics
            session_median = np.median(baseline_peaks)
            session_std = np.std(baseline_peaks)

            # --- Pass 2: Apply dynamic rejection & calculate scores ---
            for t in trials:
                # Ensure subjective_label is set (backing up original cache label)
                if "subjective_label" not in t:
                    t["subjective_label"] = t.get("label", 2)

                # Dynamically set active label based on classification config
                if USE_SUBJECTIVE_TRIAL_TYPE:
                    t["label"] = t["subjective_label"]
                else:
                    # Determine objective label from image_detail (which stores the image_type column or filename)
                    img_type = str(t.get("image_detail", t.get("image_type", ""))).lower().strip()
                    if "negative" in img_type or img_type == "1":
                        t["label"] = 1
                    elif "neutral" in img_type or img_type == "2":
                        t["label"] = 2
                    else:
                        t["label"] = t["subjective_label"]

                tw = t["times_wide"]
                ep = t["epoch_proc_wide"]
                bm = t["baseline_mean"]
                trial_max = t["baseline_max"]

                # Calculate Z-Score (handle edge case of zero variance)
                z_score = 0
                if session_std > 0:
                    z_score = (trial_max - session_median) / session_std

                # Rejection conditions
                is_absolute_outlier = trial_max > ABSOLUTE_MAX_UV
                is_dynamic_outlier = z_score > Z_SCORE_THRESHOLD

                t["rejected"] = bool(is_absolute_outlier or is_dynamic_outlier)

                score_mask = (tw >= SCORE_WIN[0]) & (tw <= SCORE_WIN[1])
                anal_mask = (tw >= ANAL_TMIN) & (tw < ANAL_TMAX)

                if t["rejected"]:
                    t["score"] = np.nan
                else:
                    t["score"] = float(np.max(ep[score_mask]) - bm)

                t["epoch_proc_anal"] = ep[anal_mask]
                t["times_anal"] = tw[anal_mask]





# ── 5. Individual trial plots ─────────────────────────────────────────────────

def plot_individual_trials(subj_data_dict, output_dir):
    """
    For each subject/session create one PNG per trial showing
    wide-window processed signal.  Rejected trials are shown in grey
    with a 'REJECTED' banner.  Accepted Negative = red, Neutral = blue.
    Also prints rejection counts per subject.
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
                    color = REJ_COLOR
                    cond_name = "REJECTED"
                else:
                    if t["label"] == 1:
                        color = NEG_COLOR
                        cond_name = "Negative"
                    else:
                        color = NEU_COLOR
                        cond_name = "Neutral"

                fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

                # Top: raw wide
                axes[0].plot(t["times_wide"] * 1000, t["epoch_raw_wide"] * 1e6,
                             color=color, linewidth=1.2, alpha=0.85)
                axes[0].axvline(0, color="k", linestyle="--", linewidth=1)
                axes[0].axvspan(-50, 0, color="gold", alpha=0.15, label="Baseline window")
                axes[0].set_ylabel("Raw (µV)")
                axes[0].set_title("Raw Signal (before filtering)")
                axes[0].legend(fontsize=8, loc="upper right")
                axes[0].grid(True, linestyle="--", alpha=0.4)

                # Bottom: processed (hp + rect + lp)
                axes[1].plot(t["times_wide"] * 1000, t["epoch_proc_wide"] * 1e6,
                             color=color, linewidth=1.5)
                axes[1].axvline(0, color="k", linestyle="--", linewidth=1)
                axes[1].axvspan(-50, 0, color="gold", alpha=0.15, label="Baseline window")
                axes[1].axvspan(SCORE_WIN[0] * 1000, SCORE_WIN[1] * 1000,
                                color="green", alpha=0.10, label="Score window (20-100 ms)")
                if not rejected:
                    axes[1].axhline(t["baseline_mean"] * 1e6, color="gray",
                                    linestyle=":", linewidth=1.2, label=f"Baseline mean")
                axes[1].set_ylabel("Processed (µV)")
                axes[1].set_xlabel("Time (ms)")
                axes[1].set_title("Processed (HP > 28 Hz → Rectify → LP < 30 Hz)")
                axes[1].legend(fontsize=8, loc="upper right")
                axes[1].grid(True, linestyle="--", alpha=0.4)

                score_str = f"{t['score']*1e6:.2f} µV" if not rejected else "REJECTED"
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


# ── 6. Trial-number scatter per subject ──────────────────────────────────────

def plot_trial_scores_per_subject(subj_data_dict, output_dir):
    """
    One figure per subject: trial number vs score, coloured by Neg/Neu.
    Rejected trials plotted as grey X markers.
    """
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

            scores = np.array([t["score"] for t in trials]) * 1e6   # µV
            labels = np.array([t["label"] for t in trials])
            rejected = np.array([t["rejected"] for t in trials])
            trial_nums = np.arange(1, len(trials) + 1)

            # Accepted
            for cond, color, name in [(1, NEG_COLOR, "Negative"), (2, NEU_COLOR, "Neutral")]:
                mask = (~rejected) & (labels == cond)
                n_trials = mask.sum()
                if n_trials > 0:
                    ax.scatter(trial_nums[mask], scores[mask], color=color,
                               s=60, edgecolors="white", linewidths=0.5, zorder=3,
                               label=f"{name} (n={n_trials})")

            # Rejected
            n_rej = rejected.sum()
            if n_rej > 0:
                ax.scatter(trial_nums[rejected], np.zeros(n_rej),
                           color=REJ_COLOR, marker="x", s=80, zorder=2,
                           label=f"Rejected (n={n_rej})")

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(f"{SESSION_MAP[sess_key]['label']}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Trial number")
            if ax == axes[0]:
                ax.set_ylabel("Score (µV – peak minus baseline)")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(plot_dir, f"{subj}_trial_scores.png")
        fig.savefig(fname, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved trial-score scatter: {fname}")


# ── 7. Group boxplot: 4 conditions ───────────────────────────────────────────

def plot_group_boxplot_four_conditions(subj_data_dict, output_dir):
    """
    Boxplot with scatter overlay:
    x-axis: Evening-Neg | Evening-Neu | Morning-Neg | Morning-Neu
    Connecting lines within subject: Eve-Neg ↔ Eve-Neu and Mor-Neg ↔ Mor-Neu.
    """
    os.makedirs(output_dir, exist_ok=True)

    cond_keys = [
        ("eve", 1, "Eve-Neg",  NEG_COLOR,  "#E8A99A"),
        ("eve", 2, "Eve-Neu",  NEU_COLOR,  "#9AC4E8"),
        ("mor", 1, "Mor-Neg",  "#922B21",  "#E8A99A"),
        ("mor", 2, "Mor-Neu",  "#1A5276",  "#9AC4E8"),
    ]

    # Collect per-subject means for each condition
    subj_means = {}   # subj -> {(sess, cond): mean_score_uv}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sess_key, cond, *_ in cond_keys:
            trials = sessions.get(sess_key)
            if not trials:
                continue
            vals = [t["score"] * 1e6 for t in trials
                    if not t["rejected"] and t["label"] == cond]
            if vals:
                subj_means[subj][(sess_key, cond)] = np.mean(vals)

    valid_subjs = [s for s in subj_means
                   if all((sk, c) in subj_means[s] for sk, c, *_ in cond_keys)]
    n_subj = len(valid_subjs)

    data_cols = []
    for sess_key, cond, *_ in cond_keys:
        data_cols.append([subj_means[s][(sess_key, cond)] for s in valid_subjs])

    n_trials_per_cond = []
    for sess_key, cond, *_ in cond_keys:
        all_trials = []
        for subj in valid_subjs:
            trials = subj_data_dict[subj].get(sess_key, [])
            all_trials += [t for t in trials if not t["rejected"] and t["label"] == cond]
        n_trials_per_cond.append(len(all_trials))

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = [1, 2, 3, 4]
    box_colors = [c[3] for c in cond_keys]
    labels_x   = [c[2] for c in cond_keys]

    bp = ax.boxplot(data_cols, positions=positions, widths=0.45,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    # Scatter + connecting lines within session (Eve: pos 1-2, Mor: pos 3-4)
    for s in valid_subjs:
        j = np.random.uniform(-0.07, 0.07)
        # Eve Neg vs Eve Neu
        v_en = subj_means[s].get(("eve", 1))
        v_eu = subj_means[s].get(("eve", 2))
        if v_en is not None and v_eu is not None:
            lc_eve = UP_COLOR if v_eu >= v_en else DOWN_COLOR
            ax.plot([1+j, 2+j], [v_en, v_eu], color=lc_eve, alpha=0.45, linewidth=1.2)
            ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(2+j, v_eu, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

        # Mor Neg vs Mor Neu
        v_mn = subj_means[s].get(("mor", 1))
        v_mu = subj_means[s].get(("mor", 2))
        if v_mn is not None and v_mu is not None:
            lc_mor = UP_COLOR if v_mu >= v_mn else DOWN_COLOR
            ax.plot([3+j, 4+j], [v_mn, v_mu], color=lc_mor, alpha=0.45, linewidth=1.2)
            ax.scatter(3+j, v_mn, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
            ax.scatter(4+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    # Statistics
    eve_neg = data_cols[0]
    eve_neu = data_cols[1]
    mor_neg = data_cols[2]
    mor_neu = data_cols[3]

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
    ax.set_ylabel("Startle Score (µV)", fontsize=11)
    ax.set_title(f"Group Startle Scores by Condition (N={n_subj} subjects)\n"
                 f"Lines connect Neg↔Neu within each session per subject",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # Divider between sessions
    ax.axvline(2.5, color="lightgray", linestyle="--", linewidth=1)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_four_conditions_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved four-condition group boxplot: {fname}")


# ── 8. Group ratio boxplot (Neg/Neu per session) ──────────────────────────────

def plot_group_ratio_boxplot(subj_data_dict, output_dir):
    """
    Boxplot + scatter: x = Evening / Morning
    y = mean(Neg score) / mean(Neu score) per subject.
    Connecting lines across sessions per subject.
    Wilcoxon signed-rank test.
    """
    os.makedirs(output_dir, exist_ok=True)

    eve_ratios, mor_ratios = [], []
    valid_subjs = []

    for subj, sessions in sorted(subj_data_dict.items()):
        ratios = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key)
            if not trials:
                continue
            neg_scores = [t["score"] * 1e6 for t in trials
                          if not t["rejected"] and t["label"] == 1]
            neu_scores = [t["score"] * 1e6 for t in trials
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
        stat, pval = stats.wilcoxon(eve_ratios, mor_ratios)
        test_label = "Wilcoxon"
    except Exception:
        stat, pval = stats.ttest_rel(eve_ratios, mor_ratios)
        test_label = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))
    bp = ax.boxplot([eve_ratios, mor_ratios], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    box_colors = [EVE_COLOR, MOR_COLOR]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for ev, mo in zip(eve_ratios, mor_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = UP_COLOR if mo >= ev else DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_ratios), max(mor_ratios)) * 1.05
    # Expand ylim so bracket fits
    ax.set_ylim(top=y_top * 1.25)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Evening\n(n={n_subj})", f"Morning\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Neg / Neu Startle Score Ratio", fontsize=11)
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


# ── 9. Group average timecourse (-50 → +120 ms) ──────────────────────────────

def plot_group_average_timecourse(subj_data_dict, output_dir):
    """
    One figure with two sub-plots (Evening | Morning).
    Each sub-plot shows the grand-average ± SEM timecourse for
    Negative (red) and Neutral (blue) accepted trials only.
    Time axis: ANAL_TMIN (−50 ms) to ANAL_TMAX (+120 ms).
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Group Average EMG Timecourse (accepted trials, −50 to +120 ms)",
                 fontsize=13, fontweight="bold")

    for ax, sess_key in zip(axes, ["eve", "mor"]):
        sess_label = SESSION_MAP[sess_key]["label"]

        neg_epochs, neu_epochs = [], []
        n_subjs_neg, n_subjs_neu = 0, 0

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key)
            if not trials:
                continue

            # Subtract per-trial baseline so the timecourse is relative to
            # pre-stimulus level (same logic as the score calculation).
            subj_neg = [(t["epoch_proc_anal"] - t["baseline_mean"]) * 1e6
                        for t in trials
                        if not t["rejected"] and t["label"] == 1]
            subj_neu = [(t["epoch_proc_anal"] - t["baseline_mean"]) * 1e6
                        for t in trials
                        if not t["rejected"] and t["label"] == 2]

            if subj_neg:
                neg_epochs.append(np.mean(subj_neg, axis=0))   # subj mean
                n_subjs_neg += 1
            if subj_neu:
                neu_epochs.append(np.mean(subj_neu, axis=0))
                n_subjs_neu += 1

        # Use the time axis from any trial in this session
        times_ms = None
        for _, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key)
            if trials:
                times_ms = trials[0]["times_anal"] * 1000   # convert to ms
                break

        if times_ms is None or (not neg_epochs and not neu_epochs):
            ax.set_title(f"{sess_label} – no data")
            continue

        for epochs, color, name, n_subjs in [
            (neg_epochs, NEG_COLOR, "Negative", n_subjs_neg),
            (neu_epochs, NEU_COLOR, "Neutral",  n_subjs_neu),
        ]:
            if not epochs:
                continue
            arr  = np.array(epochs)                        # shape: (n_subjs, n_times)
            avg  = np.mean(arr, axis=0)
            sem  = stats.sem(arr, axis=0, nan_policy="omit")
            n_trials_total = sum(
                len([t for t in sessions.get(sess_key, [])
                     if not t["rejected"] and t["label"] == (1 if name == "Negative" else 2)])
                for _, sessions in subj_data_dict.items()
            )

            ax.plot(times_ms, avg, color=color, linewidth=2.2,
                    label=f"{name} (n subj={n_subjs}, n trials={n_trials_total})")
            ax.fill_between(times_ms, avg - sem, avg + sem,
                            color=color, alpha=0.20)

        ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
        ax.axvspan(SCORE_WIN[0] * 1000, SCORE_WIN[1] * 1000,
                   color="green", alpha=0.08, label="Score window (20-100 ms)")
        ax.set_title(sess_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("EMG Amplitude (µV)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fname = os.path.join(output_dir, "group_average_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved group average timecourse: {fname}")


def plot_subject_average_timecourse(subj_data_dict, output_dir):
    """
    For each subject, create one figure with two sub-plots (Evening | Morning).
    Each sub-plot shows that subject's average ± SEM timecourse for
    Negative (red) and Neutral (blue) accepted trials only.
    """
    plot_dir = os.path.join(output_dir, "subject_average_timecourses")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        fig.suptitle(f"{subj} – Average EMG Timecourse (accepted trials, −50 to +120 ms)",
                     fontsize=13, fontweight="bold")

        has_any_data = False

        for ax, sess_key in zip(axes, ["eve", "mor"]):
            sess_label = SESSION_MAP[sess_key]["label"]
            trials = sessions.get(sess_key, [])

            if not trials:
                ax.set_title(f"{sess_label} – no data")
                continue

            # Subtract per-trial baseline
            neg_epochs = [(t["epoch_proc_anal"] - t["baseline_mean"]) * 1e6
                          for t in trials
                          if not t["rejected"] and t["label"] == 1]
            neu_epochs = [(t["epoch_proc_anal"] - t["baseline_mean"]) * 1e6
                          for t in trials
                          if not t["rejected"] and t["label"] == 2]

            times_ms = trials[0]["times_anal"] * 1000   # convert to ms

            for epochs, color, name in [
                (neg_epochs, NEG_COLOR, "Negative"),
                (neu_epochs, NEU_COLOR, "Neutral"),
            ]:
                if not epochs:
                    continue
                has_any_data = True
                arr  = np.array(epochs)                        # shape: (n_trials, n_times)
                avg  = np.mean(arr, axis=0)
                # SEM across trials for this subject
                sem  = stats.sem(arr, axis=0, nan_policy="omit") if len(epochs) > 1 else np.zeros_like(avg)

                ax.plot(times_ms, avg, color=color, linewidth=2.2,
                        label=f"{name} (n trials={len(epochs)})")
                if len(epochs) > 1:
                    ax.fill_between(times_ms, avg - sem, avg + sem,
                                    color=color, alpha=0.20)

            ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
            ax.axvspan(-50, 0, color="gold", alpha=0.12, label="Baseline window")
            ax.axvspan(SCORE_WIN[0] * 1000, SCORE_WIN[1] * 1000,
                       color="green", alpha=0.08, label="Score window (20-100 ms)")
            ax.set_title(sess_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time (ms)", fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel("EMG Amplitude (µV, baseline-corrected)", fontsize=11)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.4)

        if has_any_data:
            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_average_timecourse.png")
            fig.savefig(fname, dpi=160, bbox_inches="tight")
            print(f"  Saved subject average timecourse: {fname}")

        plt.close(fig)


def plot_subject_psd(subj_data_dict, output_dir):
    """
    For each subject, plot the Power Spectral Density (PSD) of the raw
    unfiltered signal for both Evening (orange) and Morning (purple) sessions
    on a single semilogy plot for direct comparison.
    """
    plot_dir = os.path.join(output_dir, "subject_psd")
    os.makedirs(plot_dir, exist_ok=True)

    for subj, sessions in sorted(subj_data_dict.items()):
        fig, ax = plt.subplots(figsize=(8, 5))

        has_any_data = False

        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            if not trials:
                continue

            # Get PSD from the first trial of the session
            first_t = trials[0]
            freqs = first_t.get("psd_freqs")
            psd   = first_t.get("psd_values")

            if freqs is None or psd is None:
                continue

            has_any_data = True
            color = EVE_COLOR if sess_key == "eve" else MOR_COLOR
            label = f"{SESSION_MAP[sess_key]['label']} ({first_t.get('channel', 'unknown')})"

            # Convert power from V²/Hz to µV²/Hz
            psd_uv = psd * 1e12

            ax.semilogy(freqs, psd_uv, color=color, linewidth=1.5, label=label)

        if has_any_data:
            ax.set_xlabel("Frequency (Hz)", fontsize=11)
            ax.set_ylabel("Power Spectral Density (µV²/Hz)", fontsize=11)
            ax.set_title(f"{subj} – Power Spectral Density (Raw signal before filtering)",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            # Show up to Nyquist frequency
            max_freq = max(freqs) if len(freqs) > 0 else 250
            ax.set_xlim(0, max_freq)

            plt.tight_layout()
            fname = os.path.join(plot_dir, f"{subj}_psd.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved subject PSD plot: {fname}")

        plt.close(fig)


# Helper ─────────────────────────────────────────────────────────────────────

UP_COLOR   = "#27AE60"   # green  – score went up   (Eve→Mor increase)
DOWN_COLOR = "#E74C3C"   # red    – score went down  (Eve→Mor decrease)


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


# ── 10. Session comparison boxplot (Eve vs Mor, grouped by condition) ─────────

def plot_group_session_comparison(subj_data_dict, output_dir):
    """
    Four boxes: Eve-Neg (1) | Mor-Neg (2) | Eve-Neu (3) | Mor-Neu (4)
    Connecting lines per subject: Neg pair (1-2) and Neu pair (3-4).
    Line colour = green if score went up, red if it went down.
    Wilcoxon brackets drawn inside the plot for each pair.
    """
    os.makedirs(output_dir, exist_ok=True)

    cond_keys = [
        ("eve", 1, "Eve-Neg", NEG_COLOR),
        ("mor", 1, "Mor-Neg", "#922B21"),
        ("eve", 2, "Eve-Neu", NEU_COLOR),
        ("mor", 2, "Mor-Neu", "#1A5276"),
    ]

    # Build per-subject means
    subj_means = {}
    for subj, sessions in subj_data_dict.items():
        subj_means[subj] = {}
        for sess_key, cond, *_ in cond_keys:
            trials = sessions.get(sess_key, [])
            vals = [t["score"] * 1e6 for t in trials
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

    # Scatter + coloured connecting lines
    for s in valid_subjs:
        j = np.random.uniform(-0.07, 0.07)
        # Neg pair: positions 1-2
        v_en = subj_means[s][("eve", 1)]
        v_mn = subj_means[s][("mor", 1)]
        lc_neg = UP_COLOR if v_mn >= v_en else DOWN_COLOR
        ax.plot([1+j, 2+j], [v_en, v_mn], color=lc_neg, alpha=0.55, linewidth=1.4)
        ax.scatter(1+j, v_en, color=cond_keys[0][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(2+j, v_mn, color=cond_keys[1][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

        # Neu pair: positions 3.6-4.6
        v_eu = subj_means[s][("eve", 2)]
        v_mu = subj_means[s][("mor", 2)]
        lc_neu = UP_COLOR if v_mu >= v_eu else DOWN_COLOR
        ax.plot([3.6+j, 4.6+j], [v_eu, v_mu], color=lc_neu, alpha=0.55, linewidth=1.4)
        ax.scatter(3.6+j, v_eu, color=cond_keys[2][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)
        ax.scatter(4.6+j, v_mu, color=cond_keys[3][3], edgecolors="k", s=45, zorder=4, linewidths=0.5)

    # Statistics
    neg_eve = data_cols[0]; neg_mor = data_cols[1]
    neu_eve = data_cols[2]; neu_mor = data_cols[3]
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
    # Expand ylim so brackets fit
    ax.set_ylim(top=max(y_top_neg, y_top_neu) * 1.20)
    _draw_stat_bracket(ax, 1, 2,   y_top_neg, p_neg, tl_neg)
    _draw_stat_bracket(ax, 3.6, 4.6, y_top_neu, p_neu, tl_neu)

    tick_labels = [f"{lbl}\n(n={nt})"
                   for (_, _, lbl, _), nt in zip(cond_keys, n_trials_per_cond)]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Startle Score (µV – peak minus baseline)", fontsize=11)
    ax.set_title(f"Evening vs Morning by Condition  (N={n_subj} subjects)\n"
                 f"Green line = score increased, Red line = score decreased",
                 fontsize=12, fontweight="bold")

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    # Divider between Neg pair and Neu pair
    ax.axvline(2.8, color="lightgray", linestyle="--", linewidth=1)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_session_comparison_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved session-comparison boxplot: {fname}")


# ── 11. Eve/Mor ratio per condition (Neg vs Neu) ──────────────────────────────

def plot_group_eve_mor_ratio(subj_data_dict, output_dir):
    """
    Two boxes: Neg ratio (Eve/Mor) | Neu ratio (Eve/Mor).
    Connecting lines per subject coloured green (ratio went up) / red (went down).
    Wilcoxon bracket inside the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    neg_ratios, neu_ratios, valid_subjs = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        r = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            for cond in [1, 2]:
                vals = [t["score"] * 1e6 for t in trials
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
    for patch, color in zip(bp["boxes"], [NEG_COLOR, NEU_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.55); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for neg_r, neu_r in zip(neg_ratios, neu_ratios):
        j = np.random.uniform(-0.05, 0.05)
        lc = UP_COLOR if neu_r >= neg_r else DOWN_COLOR
        ax.plot([1+j, 2+j], [neg_r, neu_r], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, neg_r, color=NEG_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, neu_r, color=NEU_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(neg_ratios), max(neu_ratios)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.axhline(1, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Negative\n(n={n_subj})", f"Neutral\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Evening / Morning Score Ratio", fontsize=11)
    ax.set_title(f"Evening-to-Morning Ratio: Neg vs Neu  (N={n_subj} subjects)\n"
                 f"Green line = ratio increased Neg→Neu, Red = decreased",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_eve_mor_ratio_boxplot.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Eve/Mor ratio boxplot: {fname}")


SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"


# ── 13. Overall Eve vs Mor boxplot (all trials, no neg/neu split) ────────────

def plot_group_overall_eve_vs_mor(subj_data_dict, output_dir):
    """
    Two boxes: Evening | Morning, using ALL accepted trials regardless of condition.
    Connecting lines per subject coloured green (↑) / red (↓).
    Wilcoxon bracket drawn inside the plot.
    """
    os.makedirs(output_dir, exist_ok=True)

    eve_means, mor_means, valid_subjs = [], [], []

    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sess_key in ["eve", "mor"]:
            trials = sessions.get(sess_key, [])
            scores = [t["score"] * 1e6 for t in trials if not t["rejected"]]
            if scores:
                vals[sess_key] = np.mean(scores)
        if "eve" in vals and "mor" in vals:
            eve_means.append(vals["eve"])
            mor_means.append(vals["mor"])
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
    for patch, color in zip(bp["boxes"], [EVE_COLOR, MOR_COLOR]):
        patch.set_facecolor(color); patch.set_alpha(0.6); patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    for ev, mo in zip(eve_means, mor_means):
        j = np.random.uniform(-0.05, 0.05)
        lc = UP_COLOR if mo >= ev else DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.6, linewidth=1.4)
        ax.scatter(1+j, ev, color=EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

    y_top = max(max(eve_means), max(mor_means)) * 1.10
    ax.set_ylim(top=y_top * 1.22)
    _draw_stat_bracket(ax, 1, 2, y_top, pval, test_label)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"Evening\n(n subj={n_subj}, n trials={n_trials_eve})",
         f"Morning\n(n subj={n_subj}, n trials={n_trials_mor})"], fontsize=10)
    ax.set_ylabel("Startle Score (µV)", fontsize=11)
    ax.set_title(f"Overall Startle Score: Evening vs Morning\n"
                 f"(N={n_subj} subjects, all accepted trials, green=↑ red=↓)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_eve_vs_mor.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall Eve vs Mor boxplot: {fname}")


# ── 14. Overall Eve/Mor ratio (all trials, tested against 1) ─────────────────

def plot_group_overall_ratio(subj_data_dict, output_dir):
    """
    One box showing Eve/Mor ratio per subject (all accepted trials).
    Dots overlaid with jitter; horizontal reference at ratio=1.
    One-sample Wilcoxon test against H0: ratio = 1.
    """
    os.makedirs(output_dir, exist_ok=True)

    ratios, valid_subjs = [], []
    for subj, sessions in sorted(subj_data_dict.items()):
        vals = {}
        for sess_key in ["eve", "mor"]:
            scores = [t["score"] * 1e6
                      for t in sessions.get(sess_key, []) if not t["rejected"]]
            if scores:
                vals[sess_key] = np.mean(scores)
        if "eve" in vals and "mor" in vals and vals["mor"] != 0:
            ratios.append(vals["eve"] / vals["mor"])
            valid_subjs.append(subj)

    n_subj = len(valid_subjs)
    if n_subj < 2:
        print("  [!] Not enough subjects for overall ratio plot")
        return

    # One-sample test: is the ratio different from 1?
    try:
        # Wilcoxon signed-rank against a constant of 1 (subtract 1 first)
        _, pval = stats.wilcoxon([r - 1 for r in ratios]); test_label = "Wilcoxon (vs 1)"
    except Exception:
        _, pval = stats.ttest_1samp(ratios, 1); test_label = "One-sample t (vs 1)"

    fig, ax = plt.subplots(figsize=(5, 6))
    bp = ax.boxplot(ratios, positions=[1], widths=0.35,
                    patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor("#95A5A6"); bp["boxes"][0].set_alpha(0.55)
    bp["boxes"][0].set_edgecolor("gray")
    bp["medians"][0].set(color="black", linewidth=2)

    for r in ratios:
        j = np.random.uniform(-0.08, 0.08)
        ax.scatter(1+j, r, color="#7F8C8D", edgecolors="k", s=60, zorder=3, linewidths=0.5)

    ax.axhline(1, color="gray", linestyle=":", linewidth=1.5)

    # Stat annotation above the box
    y_top = max(ratios) * 1.08
    ax.set_ylim(top=y_top * 1.20)
    sig = _sig_label(pval)
    ax.text(1, y_top * 1.05,
            f"{sig}\n{test_label}\np={pval:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks([1])
    ax.set_xticklabels([f"Eve / Mor\n(n={n_subj})"], fontsize=11)
    ax.set_ylabel("Evening / Morning Score Ratio", fontsize=11)
    ax.set_title("Overall Eve/Mor Ratio (all accepted trials)\n"
                 "Tested against ratio=1 (ratio > 1 = Eve > Mor)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_ratio.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall ratio plot: {fname}")


# ── 15. Overall average timecourse (Eve vs Mor, all trials) ──────────────────

def plot_group_overall_timecourse(subj_data_dict, output_dir):
    """
    Single figure with two timecourse lines: Evening (orange) and Morning (purple),
    averaged across ALL accepted trials (no neg/neu split).
    Shaded ±SEM band per session.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    sess_styles = [
        ("eve", "Evening", EVE_COLOR),
        ("mor", "Morning", MOR_COLOR),
    ]

    times_ms = None
    for sess_key, sess_label, color in sess_styles:
        subj_avgs = []
        n_subjs = 0
        n_trials_total = 0

        for subj, sessions in subj_data_dict.items():
            trials = sessions.get(sess_key, [])
            accepted = [t for t in trials if not t["rejected"]]
            if not accepted:
                continue

            # Baseline-subtract each trial before averaging
            epochs = [(t["epoch_proc_anal"] - t["baseline_mean"]) * 1e6
                      for t in accepted]
            subj_avgs.append(np.mean(epochs, axis=0))
            n_subjs += 1
            n_trials_total += len(accepted)

            if times_ms is None:
                times_ms = accepted[0]["times_anal"] * 1000

        if not subj_avgs or times_ms is None:
            continue

        arr = np.array(subj_avgs)   # (n_subjs, n_times)
        avg = np.mean(arr, axis=0)
        sem = stats.sem(arr, axis=0, nan_policy="omit")

        ax.plot(times_ms, avg, color=color, linewidth=2.5,
                label=f"{sess_label} (n subj={n_subjs}, n trials={n_trials_total})")
        ax.fill_between(times_ms, avg - sem, avg + sem, color=color, alpha=0.20)

    ax.axvline(0, color="k", linestyle="--", linewidth=1.2, label="Startle onset")
    ax.axvspan(SCORE_WIN[0]*1000, SCORE_WIN[1]*1000,
               color="green", alpha=0.08, label="Score window (20-100 ms)")
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("EMG Amplitude (µV, baseline-corrected)", fontsize=11)
    ax.set_title("Overall Average EMG Timecourse: Evening vs Morning\n"
                 "(all accepted trials, ±SEM shaded)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    fname = os.path.join(output_dir, "group_overall_timecourse.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overall average timecourse: {fname}")


# ── 16. STAI-T correlation grid ──────────────────────────────────────────────

def plot_stai_correlations(subj_data_dict, output_dir,
                           xlsx_path=SUBJECTS_XLSX):
    """
    Load STAI-T scores from the subjects spreadsheet and correlate them
    with every EMG and subjective measure.

    Measures computed per subject
    (µV for EMG, raw rating scale for subjective):
      EMG
        eve_neg, eve_neu, mor_neg, mor_neu
        neg_neu_diff  = mean(eve_neg, mor_neg) - mean(eve_neu, mor_neu)
        eve_mor_diff  = mean(eve_neg, eve_neu) - mean(mor_neg, mor_neu)
      Subjective (valence & arousal for negative trials)
        valence_eve, valence_mor, valence_diff
        arousal_eve, arousal_mor, arousal_diff
    """
    def safe_ratio(num, denom):
        if num is None or denom is None or pd.isna(num) or pd.isna(denom) or denom == 0:
            return np.nan
        return float(num) / float(denom)

    def safe_mean(lst):
        valid = [x for x in lst if not pd.isna(x) and not np.isinf(x)]
        return float(np.mean(valid)) if valid else np.nan

    # ── Load STAI-T ──
    try:
        stai_df = pd.read_excel(xlsx_path)
    except Exception as e:
        print(f"  [!] Cannot read {xlsx_path}: {e}")
        return

    stai_df["ID"] = stai_df["ID"].astype(str).str.strip().str.upper()
    stai_map = dict(zip(stai_df["ID"], stai_df["STAI-T"]))

    # ── Build per-subject measures ──
    rows = []
    for subj, sessions in sorted(subj_data_dict.items()):
        stai = stai_map.get(subj.upper())
        if stai is None or pd.isna(stai):
            continue

        m = {"subj": subj, "stai": float(stai)}

        # EMG scores
        for sess_key, cond_label in [("eve", "neg"), ("eve", "neu"),
                                      ("mor", "neg"), ("mor", "neu")]:
            cond_val = 1 if cond_label == "neg" else 2
            trials = sessions.get(sess_key, [])
            vals = [t["score"] * 1e6 for t in trials
                    if not t["rejected"] and t["label"] == cond_val]
            m[f"emg_{sess_key}_{cond_label}"] = np.mean(vals) if vals else np.nan

        # EMG ratios
        m["emg_neg_neu_ratio"] = safe_mean([
            safe_ratio(m.get("emg_eve_neg"), m.get("emg_eve_neu")),
            safe_ratio(m.get("emg_mor_neg"), m.get("emg_mor_neu"))
        ])
        m["emg_eve_mor_ratio"] = safe_mean([
            safe_ratio(m.get("emg_eve_neg"), m.get("emg_mor_neg")),
            safe_ratio(m.get("emg_eve_neu"), m.get("emg_mor_neu"))
        ])
        m["emg_neu_eve_mor_ratio"] = safe_ratio(m.get("emg_eve_neu"), m.get("emg_mor_neu"))

        # Subjective (valence & arousal for Negative trials only)
        for sess_key, label in [("eve", "eve"), ("mor", "mor")]:
            trials = sessions.get(sess_key, [])
            neg_trials = [t for t in trials
                          if not t["rejected"] and t["label"] == 1]
            m[f"val_{label}"]  = np.mean([t["valence"] for t in neg_trials]) if neg_trials else np.nan
            m[f"arsl_{label}"] = np.mean([t["arousal"] for t in neg_trials]) if neg_trials else np.nan

        m["val_ratio"]  = safe_ratio(m.get("val_eve"),  m.get("val_mor"))
        m["arsl_ratio"] = safe_ratio(m.get("arsl_eve"), m.get("arsl_mor"))

        # Subjective negative percentage
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

    # ── Define panels ──
    panels = [
        # (column,                    y-axis label,                       title)
        ("emg_eve_neg",   "EMG Score (µV)",          "Evening – Negative"),
        ("emg_eve_neu",   "EMG Score (µV)",          "Evening – Neutral"),
        ("emg_mor_neg",   "EMG Score (µV)",          "Morning – Negative"),
        ("emg_mor_neu",   "EMG Score (µV)",          "Morning – Neutral"),
        ("emg_neg_neu_ratio", "Neg / Neu Ratio",      "Neg/Neu Ratio (avg sessions)"),
        ("emg_eve_mor_ratio", "Eve / Mor Ratio",      "Eve/Mor Ratio (avg conditions)"),
        ("emg_neu_eve_mor_ratio", "Eve / Mor Neutral Ratio", "Eve/Mor Neutral Ratio"),
        ("val_eve",        "Valence rating",            "Valence – Evening (Neg trials)"),
        ("val_mor",        "Valence rating",            "Valence – Morning (Neg trials)"),
        ("val_ratio",      "Valence Ratio Eve/Mor",     "Valence Ratio Eve/Mor"),
        ("arsl_eve",       "Arousal rating",            "Arousal – Evening (Neg trials)"),
        ("arsl_mor",       "Arousal rating",            "Arousal – Morning (Neg trials)"),
        ("arsl_ratio",     "Arousal Ratio Eve/Mor",     "Arousal Ratio Eve/Mor"),
        ("neg_pct_eve",    "% Neg trials",              "% Neg trials (Evening)"),
    ]

    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.0))
    axes = axes.flatten()

    for idx, (col, ylabel, title) in enumerate(panels):
        ax = axes[idx]
        y_vals = df[col].values.astype(float)
        x_vals = df["stai"].values.astype(float)

        # Drop NaN pairs
        mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x, y = x_vals[mask], y_vals[mask]
        n = mask.sum()

        # Handle cases with constant values / no variance to prevent PearsonR/polyfit crashes
        if n < 3 or np.std(x) == 0 or np.std(y) == 0:
            ax.set_title(title + "\n(no variance / constant data)", fontsize=10)
            ax.axis("off")
            continue

        # Pearson correlation
        r, p = stats.pearsonr(x, y)
        sig = _sig_label(p)

        # Scatter
        if "neg" in col or "neg" in title.lower():
            dot_color = NEG_COLOR
        elif "neu" in col or "neu" in title.lower():
            dot_color = NEU_COLOR
        else:
            dot_color = "#7F8C8D"

        ax.scatter(x, y, color=dot_color, edgecolors="white",
                   s=60, linewidths=0.5, zorder=3, alpha=0.85)

        # Regression line
        m_coef, b_coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m_coef * x_line + b_coef,
                color="#2C3E50", linewidth=1.6, linestyle="--", zorder=2)

        # Label each subject point
        for subj_id, xi, yi in zip(df.index[mask], x, y):
            ax.annotate(subj_id, (xi, yi),
                        fontsize=6, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points",
                        color="#555555")

        # Stat box inside the plot
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        stat_txt = f"r={r:+.2f}  {p_str}  {sig}\nn={n}"
        ax.text(0.04, 0.97, stat_txt,
                transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#CCCCCC", alpha=0.85))

        ax.set_xlabel("STAI-T score", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)

    # Hide unused axes
    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("STAI-T Correlations with EMG Startle & Subjective Ratings",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = os.path.join(output_dir, "stai_correlations.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved STAI-T correlation grid: {fname}")


def plot_subjective_negative_percentage(raw_data_dir, output_dir):
    """
    Find all subjects, load their Evening and Morning ratings CSVs,
    compute the percentage of Negative trials (arousal >= 7 or valence <= 3)
    out of the sound trials, and plot them as boxplots with connecting lines.
    """
    subject_folders = sorted([
        d for d in os.listdir(raw_data_dir)
        if os.path.isdir(os.path.join(raw_data_dir, d))
    ])

    data = []
    for subj in subject_folders:
        subj_path = os.path.join(raw_data_dir, subj)
        startle_folder = find_startle_output_folder(subj_path)
        if startle_folder is None:
            continue

        pcts = {}
        for sess_key, sess_info in SESSION_MAP.items():
            csv_path = find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            if csv_path is None:
                continue

            try:
                df = load_and_classify_ratings(csv_path)
                if len(df) > 0:
                    neg_count = sum(df["subjective_label"] == 1)
                    pcts[sess_key] = 100.0 * neg_count / len(df)
            except Exception as e:
                print(f"  [!] Error reading ratings for {subj} {sess_key}: {e}")
                continue

        if "eve" in pcts and "mor" in pcts:
            data.append({
                "subj": subj,
                "eve_pct": pcts["eve"],
                "mor_pct": pcts["mor"]
            })

    n_subj = len(data)
    if n_subj < 2:
        print("  [!] Not enough subjects with rating CSVs for percentage plot")
        return

    df_pcts = pd.DataFrame(data)

    # Calculate statistics
    eve_vals = df_pcts["eve_pct"].values
    mor_vals = df_pcts["mor_pct"].values
    try:
        _, pval = stats.wilcoxon(eve_vals, mor_vals)
        test_label = "Wilcoxon"
    except Exception:
        _, pval = stats.ttest_rel(eve_vals, mor_vals)
        test_label = "Paired t"

    fig, ax = plt.subplots(figsize=(6, 6))

    # Boxplots
    bp = ax.boxplot([eve_vals, mor_vals], positions=[1, 2], widths=0.4,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [EVE_COLOR, MOR_COLOR]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("gray")
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)

    # Connecting lines and scatter
    for _, row in df_pcts.iterrows():
        ev = row["eve_pct"]
        mo = row["mor_pct"]
        j = np.random.uniform(-0.05, 0.05)
        lc = UP_COLOR if mo >= ev else DOWN_COLOR
        ax.plot([1+j, 2+j], [ev, mo], color=lc, alpha=0.55, linewidth=1.2)
        ax.scatter(1+j, ev, color=EVE_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)
        ax.scatter(2+j, mo, color=MOR_COLOR, edgecolors="k", s=55, zorder=3, linewidths=0.5)

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


# ── 17. Main ──────────────────────────────────────────────────────────────────

def main():
    # ==========================================================================
    # EXECUTION TOGGLES
    # ==========================================================================
    FORCE_PREPROCESSING             = False  # re-run from raw MFF files
    PLOT_INDIVIDUAL_TRIALS          = False  # individual trial PNGs per subject
    PLOT_TRIAL_SCORE_SCATTER        = False  # score-by-trial-number per subject
    PLOT_SUBJECT_AVERAGE_TIMECOURSE = False  # subject avg timecourse (neg/neu split)
    PLOT_SUBJECT_PSD                = False  # PSD plot per subject (unfiltered)
    PLOT_GROUP_AVERAGE_TIMECOURSE   = True   # grand-avg timecourse (neg/neu split)
    PLOT_GROUP_FOUR_CONDITIONS      = True   # group 4-condition boxplot
    PLOT_GROUP_RATIO                = True   # group Neg/Neu ratio boxplot
    PLOT_GROUP_SESSION_COMPARISON   = True   # Eve-Neg/Mor-Neg/Eve-Neu/Mor-Neu with stats
    PLOT_GROUP_EVE_MOR_RATIO        = True   # Eve/Mor ratio per condition
    PLOT_GROUP_OVERALL_TIMECOURSE   = True   # avg timecourse Eve vs Mor (all trials)
    PLOT_GROUP_OVERALL_EVE_VS_MOR   = True   # boxplot Eve vs Mor (all trials)
    PLOT_GROUP_OVERALL_RATIO        = True   # Eve/Mor ratio all trials vs 1
    PLOT_STAI_CORRELATIONS          = True   # STAI-T correlation grid
    PLOT_SUBJECTIVE_NEG_PERCENTAGE  = True   # % subjective Neg trials Eve vs Mor (CSV only)
    # ==========================================================================

    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    cache_path = os.path.join(OUTPUT_DIR, "raw_signals_cache.pkl")

    # ── Load or build raw signals cache ──
    # The cache stores ONLY signals + metadata from MFF files.
    # Rejection / score / analysis epoch are ALWAYS re-derived below
    # from the current REJECT_THRESHOLD_UV / SCORE_WIN / ANAL_TMIN/MAX.
    cache_loaded = False
    subj_data_dict = {}
    if os.path.exists(cache_path) and not FORCE_PREPROCESSING:
        print("Loading raw signals cache ...")
        try:
            with open(cache_path, "rb") as f:
                subj_data_dict = pickle.load(f)
            cache_loaded = True
        except Exception as e:
            print(f"  [!] Failed to load cache: {e}. Will rebuild.")
            subj_data_dict = {}

    subject_folders = sorted([
        d for d in os.listdir(RAW_DATA_DIR)
        if os.path.isdir(os.path.join(RAW_DATA_DIR, d))
    ])

    cache_changed = False

    for subj in subject_folders:
        subj_path   = os.path.join(RAW_DATA_DIR, subj)
        eeg_folder  = os.path.join(subj_path, "EEG")
        if not os.path.isdir(eeg_folder):
            continue

        startle_folder = find_startle_output_folder(subj_path)
        if startle_folder is None:
            continue

        # Resolve channel for this subject
        ch = SUBJECT_CHANNEL_OVERRIDES.get(subj, EMG_CHANNEL)

        # Check if we can use the cached data for this subject
        use_cache = False
        if not FORCE_PREPROCESSING and subj in subj_data_dict:
            cached_sessions = subj_data_dict[subj]
            if cached_sessions:
                # Find the first non-empty session to check channel and PSD keys
                first_sess_trials = next((t for s in cached_sessions.values() for t in s), None)
                if (first_sess_trials is not None and 
                    first_sess_trials.get("channel") == ch and 
                    "psd_freqs" in first_sess_trials):
                    use_cache = True
                else:
                    # Missing PSD data or wrong channel, force reload
                    use_cache = False

        if use_cache:
            continue

        print(f"\n--- {subj} (channel={ch}) ---")
        subj_data_dict[subj] = {}
        cache_changed = True

        for sess_key, sess_info in SESSION_MAP.items():
            mff_path = find_mff_file(eeg_folder, sess_key)
            if mff_path is None:
                print(f"  [!] No _{sess_key}_ MFF - skipping")
                continue

            csv_path = find_csv_by_suffix(startle_folder, sess_info["csv_suffix"])
            if csv_path is None:
                print(f"  [!] No {sess_info['csv_suffix']}.csv - skipping")
                continue

            print(f"  [{sess_info['label']}]")
            ratings_df = load_and_classify_ratings(csv_path)

            try:
                trials = process_session(mff_path, ratings_df, channel=ch)
            except Exception as e:
                print(f"    [!] Error: {e}")
                continue

            if trials:
                subj_data_dict[subj][sess_key] = trials

    # Save cache if we processed anything new
    if cache_changed or FORCE_PREPROCESSING or not cache_loaded:
        with open(cache_path, "wb") as f:
            pickle.dump(subj_data_dict, f)
        print("\nRaw signals cache updated.")

    # ── Always re-apply analysis parameters ──
    # This is fast (no disk I/O) and ensures rejected/score/anal epoch
    # always match the current REJECT_THRESHOLD_UV / SCORE_WIN / ANAL_TMIN/MAX.
    print("Applying analysis parameters (threshold, score window, anal window) ...")
    apply_analysis_params(subj_data_dict)

    # ── Rejection summary ──
    print("\n== Rejection Summary =============================================")
    for subj, sessions in sorted(subj_data_dict.items()):
        total_rej, total_all = 0, 0
        for sess_key, trials in sessions.items():
            total_all += len(trials)
            total_rej += sum(1 for t in trials if t["rejected"])
        pct = 100 * total_rej / max(total_all, 1)
        print(f"  {subj}: {total_all} trials | {total_rej} rejected ({pct:.0f}%)")
    print()

    # ── Plotting ──
    if PLOT_INDIVIDUAL_TRIALS:
        print("Generating individual trial plots ...")
        plot_individual_trials(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_TRIAL_SCORE_SCATTER:
        print("\nGenerating trial-score scatter plots ...")
        plot_trial_scores_per_subject(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_SUBJECT_AVERAGE_TIMECOURSE:
        print("\nGenerating subject average timecourse plots ...")
        plot_subject_average_timecourse(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_SUBJECT_PSD:
        print("\nGenerating subject PSD plots ...")
        plot_subject_psd(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_AVERAGE_TIMECOURSE:
        print("\nGenerating group average timecourse ...")
        plot_group_average_timecourse(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_FOUR_CONDITIONS:
        print("\nGenerating group four-condition boxplot ...")
        plot_group_boxplot_four_conditions(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_RATIO:
        print("\nGenerating Neg/Neu ratio boxplot ...")
        plot_group_ratio_boxplot(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_SESSION_COMPARISON:
        print("\nGenerating session-comparison boxplot ...")
        plot_group_session_comparison(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_EVE_MOR_RATIO:
        print("\nGenerating Evening/Morning ratio boxplot ...")
        plot_group_eve_mor_ratio(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_STAI_CORRELATIONS:
        print("\nGenerating STAI-T correlation grid ...")
        plot_stai_correlations(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_TIMECOURSE:
        print("\nGenerating overall avg timecourse (Eve vs Mor) ...")
        plot_group_overall_timecourse(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_EVE_VS_MOR:
        print("\nGenerating overall Eve vs Mor boxplot ...")
        plot_group_overall_eve_vs_mor(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_GROUP_OVERALL_RATIO:
        print("\nGenerating overall Eve/Mor ratio plot ...")
        plot_group_overall_ratio(subj_data_dict, PLOT_OUTPUT_DIR)

    if PLOT_SUBJECTIVE_NEG_PERCENTAGE:
        print("\nGenerating subjective negative percentage boxplot ...")
        plot_subjective_negative_percentage(RAW_DATA_DIR, PLOT_OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
