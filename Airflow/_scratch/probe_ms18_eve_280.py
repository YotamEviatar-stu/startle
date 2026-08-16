import sys
import numpy as np

sys.path.insert(0, "/Users/yotameviatar/startle-1")
sys.path.insert(0, "/Users/yotameviatar/startle-1/Airflow/_scratch")

from Airflow import airflow_config as config
from Airflow import airflow_qc as qc
import verify_spec_numbers as V

T0, T1 = 265.0, 300.0

cache = V.load_cache()
S = V.session_state(cache, "MS18", "eve")
cyc = S["win_cycles"]
med = float(np.median([c["RA"] for c in cyc]))
med_rp = float(np.median([c["RP"] for c in cyc]))
x, sf = S["signal_raw"], S["native_sfreq"]
t_raw = np.arange(len(x)) / sf
A = qc.session_breath_size(x, sf)

print(f"median RA {med:.7f}   median RP {med_rp:.2f} s   A(session_breath_size) {A:.9f}")
print(f"deep_spans dropped by flag_deep_breaths: {[(round(a,2), round(b,2)) for a, b in S['deep_spans']]}")
print(f"masked_spans from flag_excursions:       {[(round(a,2), round(b,2)) for a, b in S['masked_spans']]}")

print(f"\ncycles overlapping {T0}-{T1} s (ids = tagging view, post-deep-breath)")
print(f"{'id':>4} {'onset':>8} {'assign':>8} {'RP':>6} {'RA':>10} {'RA/med':>7} {'RFR':>9} {'peak/A':>7} {'n_pk':>4}")
for i, c in enumerate(cyc, 1):
    if c["assign_time"] < T0 or c["onset_time"] > T1:
        continue
    m = (t_raw >= c["onset_time"]) & (t_raw < c["assign_time"])
    seg = x[m]
    pk = float(np.max(np.abs(seg - np.median(x)))) / A if seg.size else float("nan")
    print(f"{i:>4} {c['onset_time']:>8.2f} {c['assign_time']:>8.2f} {c['RP']:>6.2f} "
          f"{c['RA']:>10.6f} {c['RA']/med:>7.2f} {c.get('RFR', float('nan')):>9.5f} {pk:>7.2f} {seg.size:>4}")

sub = (t_raw >= T0) & (t_raw <= T1)
pk_i = int(np.argmax(np.abs(x[sub] - np.median(x))))
pk_t = float(t_raw[sub][pk_i])
pk_v = float(x[sub][pk_i])
print(f"\nlargest |deviation| in {T0}-{T1} s: {pk_v:.8f} at t={pk_t:.2f} s = {abs(pk_v - np.median(x))/A:.2f} x A")

print("\nflag_excursions reference-window sweep at this peak (k=%.1f)" % config.QC_EXCURSION_K)
for win in (1.0, 10.0, 30.0, 60.0):
    ref = qc._rolling_median(x, int(round(win * sf)))
    dev = np.abs(x - ref)
    local = dev[sub] / A
    hit = float(np.max(local))
    frac = float(np.mean(dev > config.QC_EXCURSION_K * A)) * 100.0
    print(f"  win {win:>5.1f}s -> peak dev {hit:>6.2f} x A  {'HIT' if hit > config.QC_EXCURSION_K else 'miss':>4}"
          f"   session flagged {frac:.3f}% of samples")

spans = [(c["onset_time"], c["assign_time"]) for c in cyc
         if c["assign_time"] >= T0 and c["onset_time"] <= T1]
if spans:
    f, n = V.ss_share(S, spans)
    print(f"\ncycles in {T0}-{T1}s: {len(spans)} spans, {n} samples = {n/S['sfreq']:.1f} s, "
          f"{f*100:.1f}% of the session's sum of squares in y(RA)")

print("\ntrials whose 12 s admission window touches this region")
for k, m in enumerate(cache["MS18"]["eve"]["trials_meta"], 1):
    ct = m.get("code_time")
    if ct is None or ct < T0 - 15 or ct > T1 + 5:
        continue
    ov = V.overlapping(S, ct, 12.0)
    print(f"  trial {k:>3} code_time {ct:>8.2f}  breaths in window: "
          f"{[i for i, _ in ov]}  RA/med {[round(c['RA']/med, 2) for _, c in ov]}")
