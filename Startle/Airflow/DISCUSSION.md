Phase 1 — filter+downsample: cached 25Hz signal_raw → 1st-order Butterworth lowpass (5Hz) → resampled to 10Hz → raw_z. used for cycle-detection only, and time frames.

Phase 2 — cycle detection: inspiration onsets found on raw_z (bandpass 0.01–0.6Hz + Hampel outlier flagging). Onset timing comes from raw_z, but RP/RA/RFR magnitude for each cycle is measured back on the unfiltered signal_raw — mirrors PsPM's pspm_resp_pp.m split between its newresp (detection) and resp (amplitude) variables. 

Phase 3 — continuous series: points are linearly interpulated to produces RP_series, RA_series, RFR_series spanning the full session.

[Open image](./image.png)

Phase 4 — per-trial GLM fit: for each trial's full window

Runs once per metric (RP, RA, RFR) per trial → score_rp, score_ra, score_rfr.


glm model =  Y=Xβ+e
y = time sereis data
x = design matrix [(t), (CRF(t)), (dCRF/dt)]
β = vector of weights, trying to match x to y
e = error

pinv:
β^=X^+Y

expected β's:

Replication of Effects: All three measures confirmed an overall event-related response, characterized by a negative RP response (breathing deceleration) and an increase in RA and RFR

review:
1. view baseline changing over time 
2. if the gaussian function includes -5
3. score
4. t=0 places in differenet phase every time.
5. gaussion from pi = 0 (bx)
6. refractory preiod.
7. latency 
8. adding a sound marker
9. image generation -
## Discussion - implementation,  less logical
1. Lineraity - authors used 40s, suggest removing linearity component for shorter windows 
2. More manual inspection needs to be done
3. No usage of pre-image data - 18 minutes simetimes - 1 minute baseline pre post
4. No confidence in cycle
5. Airflow vs chest belt
6. PsPM defult setting - latnecy = fixed/free
7. adapting to variability for each subject
8. matlab to python
9. difference  
10. shoulder before peak
11. noisy baseline can be removed
---

## Problems — concrete mismatches/gaps to fix

1. Our **cycle-detection** filter is a single 2nd-order bandpass where PsPM runs two separate 1st-order `filtfilt` passes (different roll-off/phase). *(The other half of this — one shared FINAL high-pass across all three measures — is now fixed; see Resolved.)*
2. Per-breath amplitude plausibility — RESOLVED as a two-layer scheme, not a single either/or choice: `SUBJECTS_EXCLUDE` (the coarse session/subject list) is now ALWAYS honoured regardless of `GLM_ARTIFACT_METHOD`, because the Hampel bound is session-relative and structurally misses a *uniformly*-corrupted session (e.g. DA01). `GLM_ARTIFACT_METHOD` (default `hampel_reject_trials`) then handles per-trial/per-cycle spikes in every session the manual list doesn't park. See `GLM_ARTIFACT_METHOD` in `airflow_config.py` for the full rationale.
3. We fit each trial separately; PsPM fits one model per condition across the whole session — a deliberate choice, but must be disclosed whenever citing the method. *(Now the leading suspect for the null GLM result — see GLM_METHOD_FOUNDATIONS.md §7.)*

---