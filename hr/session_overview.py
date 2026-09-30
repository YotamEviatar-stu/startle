import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hr import config as cfg
from hr.ica_cardiac import OUT_DIR as HR_DIR, find_mff_file, session_spans, subject_folder

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out_overview")

REST = ("rest_pre", "rest_post")
WCOL = {"rest_pre": "#2f9e44", "rest_post": "#0c8599", "task": "#f4d3c2", "other": "0.65"}
FULL_BINS = 6000
ZOOM_PAD_S = 1.0
HRV_WIN_S = 10.0
HRV_MIN_DIFFS = 5


def successive_diffs(beat_t, ok):
    ibi = np.diff(beat_t)
    good = ok[:-1] & ok[1:]
    return beat_t[1:-1][good], np.diff(ibi)[good] * 1000.0


def rmssd(d):
    return float(np.sqrt(np.mean(d ** 2))) if d.size >= HRV_MIN_DIFFS else np.nan


def in_bad(a, b, bad):
    return any(a < hi and b > lo for lo, hi in bad)


def tile(span, t, d, bad, label_of):
    out = []
    lo, hi = span
    for i in range(int((hi - lo) // HRV_WIN_S)):
        a = lo + i * HRV_WIN_S
        b = a + HRV_WIN_S
        if in_bad(a, b, bad):
            continue
        m = (t >= a) & (t < b)
        out.append((a, rmssd(d[m]), label_of(a, b)))
    return [w for w in out if np.isfinite(w[1])]


def compute_session(subj, sess, status):
    key = f"{subj}/{sess}"
    out = dict(subj=subj, sess=sess, key=key, status=status)
    npz = os.path.join(HR_DIR, f"{subj}_{sess}.npz")
    if not status.startswith("ok") or not os.path.exists(npz):
        return out
    z = np.load(npz, allow_pickle=True)
    out["hr_source"] = str(z["hr_source"])
    mff = find_mff_file(os.path.join(cfg.RAW_DATA_DIR, subject_folder(subj), "EEG"), sess)
    spans = session_spans(mff, key)
    out["spans"] = spans
    out["bad"] = cfg.MANUAL_BAD_SPANS.get(key, [])
    if "beat_times_s" not in z:
        return out

    beat_t, ok = z["beat_times_s"].astype(float), z["ibi_ok"].astype(bool)
    t, d = successive_diffs(beat_t, ok)
    keep = np.array([not in_bad(x, x, out["bad"]) for x in t], dtype=bool) if out["bad"] else np.ones(t.size, bool)
    t, d = t[keep], d[keep]

    def label_of(a, b):
        for name in REST:
            s = spans[name]
            if s is not None and a >= s[0] and b <= s[1]:
                return name
        return "other"

    stage = {}
    rest_windows = []
    for name in REST:
        if spans[name] is None:
            stage[name] = dict(rmssd=np.nan, n_beats=0, n_windows=0)
            continue
        lo, hi = spans[name]
        m = (t >= lo) & (t < hi)
        w = tile(spans[name], t, d, out["bad"], lambda a, b, n=name: n)
        rest_windows += w
        stage[name] = dict(rmssd=rmssd(d[m]), n_beats=int(((beat_t >= lo) & (beat_t < hi)).sum()),
                           n_windows=len(w))
    full_windows = tile(spans["full"], t, d, out["bad"], label_of)

    ibi = np.diff(beat_t)
    ibi_t = beat_t[:-1] + ibi / 2.0
    ibi_bpm = 60.0 / ibi

    def part(lo, hi, windows):
        m = (ibi_t >= lo) & (ibi_t < hi)
        w = [x[1] for x in windows if x[0] >= lo and x[0] + HRV_WIN_S <= hi]
        return dict(kept=int((m & ok).sum()), excluded=int((m & ~ok).sum()),
                    median_bpm=float(np.median(ibi_bpm[m & ok])) if (m & ok).any() else np.nan,
                    median_hrv=float(np.median(w)) if w else np.nan)

    parts = {"full": part(*spans["full"], full_windows)}
    rest_mask = np.zeros(ibi.size, bool)
    rest_lab = np.array([""] * ibi.size, dtype=object)
    for name in REST:
        if spans[name] is not None:
            parts[name] = part(*spans[name], rest_windows)
            m = (ibi_t >= spans[name][0]) & (ibi_t < spans[name][1]) & ok
            rest_mask |= m
            rest_lab[m] = name

    out.update(beat_t=beat_t, ok=ok, ibi_bpm=ibi_bpm, parts=parts, rest_mask=rest_mask,
               rest_lab=rest_lab, stage=stage, rest_windows=rest_windows, full_windows=full_windows,
               fix_median_bpm=float(np.median(ibi_bpm[rest_mask])) if rest_mask.any() else np.nan,
               fix_median=float(np.median([w[1] for w in rest_windows])) if rest_windows else np.nan,
               full_median=float(np.median([w[1] for w in full_windows])) if full_windows else np.nan,
               src=z["primary_source"].astype(float), src_sr=float(z["source_srate"]),
               src_t0=float(z["source_t0"]))
    return out


def minmax_envelope(t, y, n_bins):
    if y.size <= 2 * n_bins:
        return t, y
    edges = np.linspace(0, y.size, n_bins + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        idx += sorted((a + int(np.argmin(seg)), a + int(np.argmax(seg))))
    idx = np.array(idx)
    return t[idx], y[idx]


def fmt(v, unit):
    return "n/a" if not np.isfinite(v) else "%.0f %s" % (v, unit)


def plot_full_session(ax, d):
    lo, hi = d["spans"]["full"]
    sp = d["spans"]
    ts = d["src_t0"] + np.arange(d["src"].size) / d["src_sr"]
    m = (ts >= lo) & (ts <= hi)
    et, ey = minmax_envelope(ts[m], d["src"][m], FULL_BINS)
    if sp["rest_pre"] is not None:
        ax.axvspan(*sp["rest_pre"], color=WCOL["rest_pre"], alpha=0.3, lw=0)
    ax.axvspan(*sp["task"], color=WCOL["task"], alpha=0.45, lw=0)
    if sp["rest_post"] is not None:
        ax.axvspan(*sp["rest_post"], color=WCOL["rest_post"], alpha=0.3, lw=0)
    for a, b in d["bad"]:
        ax.axvspan(a, b, color="crimson", alpha=0.35, lw=0)
    ax.plot(et, ey, "k", lw=0.5)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    p = d["parts"]
    kept = lambda n: p[n]["kept"] if n in p else 0
    ax.set_title("%s full session  |  rest_pre kept %d, rest_post kept %d  |  beats kept %d, excluded %d  |  "
                 "median HR %s  |  median HRV %s  |  %s"
                 % (d["key"], kept("rest_pre"), kept("rest_post"), p["full"]["kept"], p["full"]["excluded"],
                    fmt(p["full"]["median_bpm"], "bpm"), fmt(p["full"]["median_hrv"], "ms"), d["hr_source"]),
                 fontsize=10, loc="left", fontweight="bold")
    ax.set_ylabel("cardiac IC (a.u.)")
    ax.set_xlabel("time (s)")


def plot_rest_zoom(ax, d, name):
    span = d["spans"][name]
    if span is None:
        ax.axis("off")
        ax.text(0.5, 0.5, f"no {name} block in this recording", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        return
    t0, t1 = span
    i0 = max(int((t0 - ZOOM_PAD_S - d["src_t0"]) * d["src_sr"]), 0)
    i1 = min(int((t1 + ZOOM_PAD_S - d["src_t0"]) * d["src_sr"]), d["src"].size)
    ts = d["src_t0"] + np.arange(i0, i1) / d["src_sr"]
    ys = d["src"][i0:i1]
    ax.axvspan(t0, t1, color=WCOL[name], alpha=0.45, lw=0)
    ax.plot(ts, ys, "k", lw=0.7)
    if ts.size:
        bt = d["beat_t"]
        sel = (bt >= ts[0]) & (bt <= ts[-1])
        good = np.zeros(bt.size, bool)
        good[:-1] |= d["ok"]
        good[1:] |= d["ok"]
        for mask, col in ((sel & good, "#c92a2a"), (sel & ~good, "0.55")):
            if mask.any():
                ax.plot(bt[mask], np.interp(bt[mask], ts, ys), "v", color=col, ms=5, zorder=4)
    for a, bb in d["bad"]:
        ax.axvspan(max(a, t0 - ZOOM_PAD_S), min(bb, t1 + ZOOM_PAD_S), color="crimson", alpha=0.08, lw=0)
    ax.set_xlim(t0 - ZOOM_PAD_S, t1 + ZOOM_PAD_S)
    ax.set_ylabel("cardiac IC (a.u.)")
    p = d["parts"][name]
    ax.set_title("%s (%.0f–%.0fs): %d kept, %d excluded  |  median HR %s  |  median HRV %s"
                 % (name, t0, t1, p["kept"], p["excluded"], fmt(p["median_bpm"], "bpm"),
                    fmt(p["median_hrv"], "ms")), fontsize=9.5, loc="left", fontweight="bold")


def plot_fixation_median(ax, d):
    vals, lab = d["ibi_bpm"][d["rest_mask"]], d["rest_lab"][d["rest_mask"]]
    order = np.argsort(vals)
    sv, lv = vals[order], lab[order]
    xs = np.arange(sv.size)
    ax.scatter(xs, sv, c=[WCOL[w] for w in lv], s=30, zorder=4, edgecolor="0.3", linewidth=0.5)
    if sv.size:
        mid = sorted({int(np.floor((sv.size - 1) / 2)), int(np.ceil((sv.size - 1) / 2))})
        ax.plot(xs[mid], sv[mid], "o", mfc="none", mec="crimson", mew=2, ms=12, zorder=5)
        ax.axhline(d["fix_median_bpm"], color="crimson", ls="--", lw=1.3)
    ax.text(0.02, 0.94, "median = %s  |  median HRV = %s  (%d × %.0f-s RMSSD windows)"
            % (fmt(d["fix_median_bpm"], "bpm"), fmt(d["fix_median"], "ms"), len(d["rest_windows"]), HRV_WIN_S),
            color="crimson", fontsize=11, fontweight="bold", transform=ax.transAxes, va="top")
    ax.set_xlabel("beats sorted by HR (bpm) (n=%d, rest_pre+rest_post pooled)" % sv.size)
    ax.set_ylabel("HR (bpm)")
    ax.set_title("Fixation median, HR (bpm) and HRV (RMSSD)", fontsize=10.5, loc="left", fontweight="bold")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", mfc=WCOL[n], mec="0.3", label=f"{n} beat")
                       for n in REST], loc="lower right", fontsize=8.5, frameon=False)


def plot_message(d, msg):
    fig = plt.figure(figsize=(20, 10.5))
    fig.text(0.5, 0.55, msg, ha="center", va="center", fontsize=14, wrap=True)
    fig.suptitle("HRV session overview (%s)" % d["key"], fontsize=13, y=0.995, fontweight="bold")
    return fig


def plot_session(d):
    if not d["status"].startswith("ok"):
        return plot_message(d, "Not processed.\n%s" % d["status"])
    if "beat_t" not in d:
        return plot_message(d, "ICA ran, but no component met the five CARACAS criteria "
                               "(no ECG-like heartbeat component),\nso this session has no HR or HRV.")
    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.1], hspace=0.7, wspace=0.22)
    plot_full_session(fig.add_subplot(gs[0, :]), d)
    plot_rest_zoom(fig.add_subplot(gs[1, 0]), d, "rest_pre")
    plot_rest_zoom(fig.add_subplot(gs[1, 1]), d, "rest_post")
    plot_fixation_median(fig.add_subplot(gs[2, :]), d)
    fig.suptitle("HR/HRV session overview (%s, cardiac ICA component)" % d["key"], fontsize=13, y=0.995,
                 fontweight="bold")
    return fig


def available_sessions():
    summary = pd.read_csv(os.path.join(HR_DIR, "summary.csv"))
    rows = []
    for _, r in summary.iterrows():
        subj, sess = r["session"].split("/")
        rows.append((subj, sess, str(r["status"])))
    return sorted(rows, key=lambda k: (k[0], k[1] != "eve"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=None, help="e.g. MS18/eve")
    ap.add_argument("--pdf", default=os.path.join(OUT_DIR, "hrv_session_overview_all_sessions.pdf"))
    a = ap.parse_args()

    keys = available_sessions()
    os.makedirs(OUT_DIR, exist_ok=True)
    table = []
    if a.session:
        wanted = set(a.session)
        for subj, sess, status in (k for k in keys if f"{k[0]}/{k[1]}" in wanted):
            fig = plot_session(compute_session(subj, sess, status))
            out = os.path.join(OUT_DIR, "%s_%s.png" % (subj, sess))
            fig.savefig(out, dpi=115, bbox_inches="tight")
            plt.close(fig)
            print("saved", out)
        return

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(a.pdf) as pdf:
        for subj, sess, status in keys:
            d = compute_session(subj, sess, status)
            fig = plot_session(d)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            st = d.get("stage", {})
            table.append({"session": d["key"], "status": status, "hr_source": d.get("hr_source", ""),
                          "rmssd_rest_pre_ms": st.get("rest_pre", {}).get("rmssd", np.nan),
                          "rmssd_rest_post_ms": st.get("rest_post", {}).get("rmssd", np.nan),
                          "fixation_median_bpm": d.get("fix_median_bpm", np.nan),
                          "fixation_median_rmssd_ms": d.get("fix_median", np.nan),
                          "full_session_median_bpm": d.get("parts", {}).get("full", {}).get("median_bpm", np.nan),
                          "full_session_median_rmssd_ms": d.get("full_median", np.nan)})
    pd.DataFrame(table).to_csv(os.path.join(OUT_DIR, "hrv_summary.csv"), index=False)
    print("saved", a.pdf)


if __name__ == "__main__":
    main()
