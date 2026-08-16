"""Follow-up to probe_peak_gates.py: RA/RP profile of the n_peaks classes,
relative to each session's own median breath. Read-only."""

import sys
import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config          # noqa: E402
from probe_peak_gates import load_cache, session_cycles  # noqa: E402


def main():
    cache = load_cache()
    excl = getattr(config, "SUBJECTS_EXCLUDE", {})
    rows = []          # (n_peaks, RP, RA_ratio, subj, sess, onset)

    for subj in sorted(cache):
        if subj in excl:
            continue
        for sess_key, sess in sorted(cache[subj].items()):
            if f"{subj}/{sess_key}" in excl:
                continue
            cycles, rejected, reason, report, _ = session_cycles(sess, subj, sess_key)
            med = report["median_RA"]
            if not (med > 0):
                continue
            for c, r in zip(cycles, reason):
                rows.append((int(c.get("n_peaks", -1)), c["RP"], c["RA"] / med,
                             r, subj, sess_key, c["onset_time"]))

    arr_np = np.array([r[0] for r in rows])
    arr_rp = np.array([r[1] for r in rows])
    arr_ra = np.array([r[2] for r in rows])

    print(f"\n{'n_peaks':>8} {'n':>6} {'RA/med p5':>10} {'p50':>8} {'p95':>8} "
          f"{'RP p50':>8} {'RP p95':>8}")
    for k in [0, 1, 2, 3]:
        m = (arr_np == k) if k < 3 else (arr_np >= 3)
        if not m.any():
            continue
        lbl = f">={k}" if k == 3 else str(k)
        print(f"{lbl:>8} {m.sum():>6} {np.percentile(arr_ra[m],5):10.4f} "
              f"{np.percentile(arr_ra[m],50):8.4f} {np.percentile(arr_ra[m],95):8.4f} "
              f"{np.percentile(arr_rp[m],50):8.2f} {np.percentile(arr_rp[m],95):8.2f}")

    z = [r for r in rows if r[0] == 0]
    ra0 = np.array([r[2] for r in z])
    print(f"\nn_peaks==0 (n={len(z)}): RA/median_RA  "
          f"min={ra0.min():.5f}  p25={np.percentile(ra0,25):.4f}  "
          f"med={np.median(ra0):.4f}  p75={np.percentile(ra0,75):.4f}  max={ra0.max():.4f}")
    print(f"  fraction under 0.10x the session median breath: "
          f"{100*np.mean(ra0 < 0.10):.1f}%   under 0.25x: {100*np.mean(ra0 < 0.25):.1f}%"
          f"   over 0.50x: {100*np.mean(ra0 > 0.50):.1f}%")

    print("\n  n_peaks==0 with the LARGEST amplitude (these look like real breaths):")
    for r in sorted(z, key=lambda x: -x[2])[:10]:
        print(f"    {r[4]}/{r[5]:<4} t={r[6]:8.2f}s  RP={r[1]:5.2f}s  RA={r[2]:6.3f}x median")

    print("\n  n_peaks==0 with the SMALLEST amplitude (dead signal):")
    for r in sorted(z, key=lambda x: x[2])[:10]:
        print(f"    {r[4]}/{r[5]:<4} t={r[6]:8.2f}s  RP={r[1]:5.2f}s  RA={r[2]:6.4f}x median")

    t2 = [r for r in rows if r[0] == 2]
    print(f"\nn_peaks==2 (n={len(t2)}, currently KEPT) — the ones a threshold of 2 would cut:")
    for r in sorted(t2, key=lambda x: -x[1])[:10]:
        print(f"    {r[4]}/{r[5]:<4} t={r[6]:8.2f}s  RP={r[1]:5.2f}s  RA={r[2]:6.3f}x median")
    rp2 = np.array([r[1] for r in t2])
    rp1 = arr_rp[arr_np == 1]
    print(f"    RP: n_peaks==2 median {np.median(rp2):.2f}s vs n_peaks==1 median "
          f"{np.median(rp1):.2f}s  ({np.median(rp2)/np.median(rp1):.2f}x)")


if __name__ == "__main__":
    main()
