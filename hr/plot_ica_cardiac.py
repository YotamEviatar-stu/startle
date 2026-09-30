import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr.ica_cardiac import OUT_DIR, FIT_BAND, removed_signal, session_key

WIN = (360.0, 365.0)
ZOOM = (360.0, 370.0)
TREND = (300.0, 420.0)
LAG_S = 3
N_AVG_BEATS = 4


def four_beat_bpm(beat_t, ok, grid):
    ibi = np.diff(beat_t)
    b4 = np.full(ibi.size, np.nan)
    for i in range(N_AVG_BEATS - 1, ibi.size):
        if ok[i - N_AVG_BEATS + 1:i + 1].all():
            b4[i] = 60.0 / ibi[i - N_AVG_BEATS + 1:i + 1].mean()
    out = np.full(grid.size, np.nan)
    j = np.searchsorted(beat_t[1:], grid, side="right") - 1
    out[j >= 0] = b4[j[j >= 0]]
    return out


def stacked(ax, t, rows, labels, colors, gap):
    for k, (y, lab, c) in enumerate(zip(rows, labels, colors)):
        off = -k * gap
        ax.plot(t, y + off, lw=0.7, color=c)
        ax.text(t[0] - 0.05 * (t[-1] - t[0]), off, lab, ha="right", va="center", fontsize=8, color=c)
    ax.set_yticks([])
    ax.set_xlim(t[0], t[-1])


def main(mff):
    subj, sess = session_key(mff)
    z = np.load(os.path.join(OUT_DIR, f"{subj}_{sess}.npz"))
    t0 = float(z["task_window"][0])
    cardiac = [int(c) for c in z["cardiac"]]
    seed = int(z["seed"])
    other = [c for c in cardiac if c != seed][0]

    removed, raw, ica = removed_signal(mff)
    sr = raw.info["sfreq"]
    picks = [raw.ch_names.index(c) for c in ica.ch_names]
    rms = removed[picks].std(1)
    show_ch = ica.ch_names[int(np.argmax(rms))]
    ci = raw.ch_names.index(show_ch)
    zl, zh = int(ZOOM[0] * sr), int(ZOOM[1] * sr)
    e_raw = raw.get_data(picks=[ci])[0][zl:zh]
    e_rem = removed[ci][zl:zh]
    del removed

    fit = raw.copy().filter(*FIT_BAND, verbose="error")
    del raw
    src = ica.get_sources(fit).get_data()
    mixing = ica.get_components()

    top = [int(i) for i in np.argsort(mixing[:, [seed, other]].__abs__().max(1))[::-1]
           if ica.ch_names[i] != show_ch][:2]
    chans = [show_ch] + [ica.ch_names[i] for i in top] + ["E36", "E18", "E116"]
    chans = [c for c in dict.fromkeys(chans) if c in ica.ch_names][:6]

    wl, wh = int(WIN[0] * sr), int(WIN[1] * sr)
    tw = np.arange(wl, wh) / sr
    eeg_rows = [fit.get_data(picks=[c])[0][wl:wh] * 1e6 for c in chans]
    comp_show = [seed, other] + [c for c in range(4) if c not in cardiac][:3]
    comp_rows = [src[c, wl:wh] / np.std(src[c]) for c in comp_show]

    beats = z["beat_times_s"] - t0
    bs = np.round(beats * sr).astype(int)
    half = int(0.6 * sr)
    bs = bs[(bs - half >= 0) & (bs + half < src.shape[1])]
    lock_t = np.arange(-half, half + 1) / sr
    lock = {c: np.mean([src[c, b - half:b + half + 1] for b in bs], axis=0) / np.std(src[c])
            for c in [seed, other]}
    null_c = comp_show[2]
    lock[null_c] = np.mean([src[null_c, b - half:b + half + 1] for b in bs], axis=0) / np.std(src[null_c])

    grid = z["t_1hz"] - t0
    pulse = z["pulse_bpm_1hz"]
    bpm4 = four_beat_bpm(z["beat_times_s"], z["ibi_ok"], z["t_1hz"])
    shifted = np.full(pulse.size, np.nan)
    shifted[:pulse.size - LAG_S] = pulse[LAG_S:]
    m = np.isfinite(bpm4) & np.isfinite(shifted)
    r = np.corrcoef(bpm4[m], shifted[m])[0, 1]
    mad = np.median(np.abs(bpm4[m] - shifted[m]))

    fig = plt.figure(figsize=(17, 22))
    gs = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    stacked(ax, tw, eeg_rows, chans, ["0.2"] * len(chans), gap=60)
    ax.set_title("STEP 1 - What the electrodes record (5 s, 1-40 Hz)\n"
                 "each channel = a MIX of brain + eyes + muscle + heart", fontsize=11, loc="left")
    ax.set_xlabel("s from D101")

    ax = fig.add_subplot(gs[0, 1])
    colors = ["C3", "C1"] + ["C0"] * 3
    labs = [f"IC{c}" + (" (pulse)" if c == seed else " (heart spike)" if c == other else "") for c in comp_show]
    stacked(ax, tw, comp_rows, labs, colors, gap=6)
    ax.set_title(f"STEP 2 - ICA unmixes {len(ica.ch_names)} channels into {ica.n_components_} independent components\n"
                 "same 5 s; two of them beat once per heartbeat", fontsize=11, loc="left")
    ax.set_xlabel("s from D101")

    ax = fig.add_subplot(gs[1, 0])
    match = z["seed_match"]
    ax.bar(np.arange(match.size), match,
           color=["C3" if i == seed else "C1" if i == other else "0.7" for i in range(match.size)])
    ax.set_xlabel("component")
    ax.set_ylabel("fraction of 10-s windows")
    ax.set_title("STEP 3 - Which component is the heart?\n"
                 f"its rhythm matches SpO2-Pulse within +/-5 bpm: IC{seed} = {match[seed]:.0%}, next best = {np.sort(match)[-2]:.0%}",
                 fontsize=11, loc="left")

    sub = gs[1, 1].subgridspec(1, 2)
    for k, c in enumerate([seed, other]):
        ax = fig.add_subplot(sub[0, k])
        mne.viz.plot_topomap(mixing[:, c], ica.info, axes=ax, show=False, contours=4)
        ax.set_title(f"IC{c} weight on each electrode\n" + ("pulse" if c == seed else "heart spike"), fontsize=10)

    ax = fig.add_subplot(gs[2, 0])
    s36 = src[seed, zl:zh] / np.std(src[seed])
    tz = np.arange(zl, zh) / sr
    ax.plot(tz, s36, color="C3", lw=0.8)
    bz = beats[(beats >= ZOOM[0]) & (beats < ZOOM[1])]
    yb = np.interp(bz, tz, s36)
    ax.plot(bz, yb, "kv", ms=7)
    for a, b in zip(bz[:-1], bz[1:]):
        ax.annotate("", xy=(b, yb.max() + 0.6), xytext=(a, yb.max() + 0.6),
                    arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((a + b) / 2, yb.max() + 0.9, f"{b - a:.2f}s\n{60 / (b - a):.0f}bpm", ha="center", fontsize=7)
    ax.set_ylim(s36.min() - 0.5, yb.max() + 2.2)
    ax.set_xlabel("s from D101")
    ax.set_title(f"STEP 4 - Find each beat on IC{seed}; gap between beats -> heart rate\n"
                 "bpm = 60 / gap", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[2, 1])
    sel = (grid >= TREND[0]) & (grid < TREND[1])
    ax.plot(grid[sel], z["bpm_1hz"][sel], color="C3", lw=0.6, alpha=0.5, label="ICA, beat-to-beat")
    ax.plot(grid[sel], bpm4[sel], color="C3", lw=1.6, label="ICA, 4-beat average")
    ax.plot(grid[sel], shifted[sel], color="k", lw=1.4, label=f"SpO2-Pulse (shifted {LAG_S} s earlier)")
    ax.set_ylabel("bpm")
    ax.set_xlabel("s from D101")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"STEP 5 - Check against the oximeter (whole session: r = {r:.2f}, median |diff| = {mad:.1f} bpm)\n"
                 "oximeter averages 4 beats and reports late, so ICA is averaged the same way", fontsize=11, loc="left")

    ax = fig.add_subplot(gs[3, 0])
    for c, col, lab in [(seed, "C3", "pulse"), (other, "C1", "heart spike"), (null_c, "0.6", "non-cardiac, for contrast")]:
        ax.plot(lock_t * 1000, lock[c], color=col, lw=1.4, label=f"IC{c} ({lab})")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("ms relative to detected beat (IC pulse peak)")
    ax.set_ylabel("average, SD units")
    ax.legend(fontsize=8)
    ax.set_title(f"STEP 6 - Average every component around the {bs.size} beats\n"
                 "cardiac ones show the same shape every beat; the electrical spike comes before the pulse",
                 fontsize=11, loc="left")

    ax = fig.add_subplot(gs[3, 1])
    tz_ = np.arange(zl, zh) / sr
    ax.plot(tz_, e_raw * 1e6, color="0.6", lw=0.7, label="raw")
    ax.plot(tz_, (e_raw - e_rem) * 1e6, color="C0", lw=0.7, label="cleaned = raw - cardiac")
    ax.plot(tz_, e_rem * 1e6 + np.mean(e_raw) * 1e6 - 60, color="C3", lw=0.9, label="cardiac signal kept (shifted down)")
    for b in bz:
        ax.axvline(b, color="k", lw=0.3, alpha=0.4)
    ax.set_xlabel("s from D101")
    ax.set_ylabel("uV")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"STEP 7 - Subtract IC{seed} + IC{other} from the EEG ({show_ch}, 10 s)\n"
                 "what is subtracted is the heart signal we keep", fontsize=11, loc="left")

    fig.suptitle(f"{subj}/{sess}: extracting the heartbeat from EEG with ICA", fontsize=14, y=0.995)
    out = os.path.join(OUT_DIR, f"{subj}_{sess}_ica_explained.png")
    fig.savefig(out, dpi=65, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "local", "MS18_eve_20251120_103853.mff"))
