"""
airflow_config.py — all tunable parameters for the Airflow (respiration) pipeline.

Two-layer cache model (referenced throughout this file):
  Layer 1 = the raw signal cache (built from the .mff files, stored at
            CACHE_SFREQ). Changing a Layer-1 setting requires FORCE_RELOAD
            (re-reads the raw recordings — slow).
  Layer 2 = everything computed from that cache (filtering, cycle detection,
            scoring, rejection gates). Free to tune and re-run — no reload.

Two scoring paths, selected by SCORING_METHOD below:
  "peak_excursion_normalized" (BxB)   — breath-by-breath peak-picking.
                                          Runs in airflow_amp_processor.py.
  "glm_deconvolution"          (GLM)  — canonical-response-function GLM fit
                                          per trial. Runs in airflow_glm.py.
Each path has its own block of settings and its own rejection gates below,
kept at parity (same names/thresholds/intent) so switching methods doesn't
silently change how strict the pipeline is.
"""

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_DIR = r"/Volumes/My Passport/startle_raw"
OUTPUT_DIR   = r"/Users/yotameviatar/Desktop/untitled folder/airflow_output"

# ── Subject filter ─────────────────────────────────────────────────────────────
SUBJECT_FILTER = []   # empty = all subjects; a non-empty list is a whitelist
                       # (debug: show only these subjects), NOT an exclusion.

# ── Subject / session exclusions [Layer 2 — free to change, no FORCE_RELOAD] ───
# DISCUSSION GOAL: agree a principled, consistent amplitude-outlier rejection rule -- this list is a placeholder, not that rule.
#
# - ALWAYS active, regardless of GLM_ARTIFACT_METHOD below. It parks whole-session
#   contamination (e.g. DA01's session-wide saturation) that the hampel_* methods
#   structurally cannot catch -- their bound is relative to each session's OWN
#   median/MAD, so a UNIFORMLY corrupted session (every cycle huge -> median huge
#   too) never trips it. GLM_ARTIFACT_METHOD controls only the per-trial/per-cycle
#   handling applied to sessions NOT parked here.
# - Keys may be a whole subject ("ER23", parks BOTH sessions) OR a single
#   session ("ER23/eve", parks only that session -- the finer granularity the
#   "fine with excluding a whole session for now" decision calls for). Current
#   entries are whole-subject; narrow one to "SUBJ/eve"/"SUBJ/mor" to keep the
#   cleaner session.
# - Parks the session(s) out of ALL scoring/plots; NOT the per-trial gates below, NOT SUBJECT_FILTER above (that's a debug whitelist)
# - Chosen by eyeballing the 2026-07-05 outlier scan + raw .mff traces (mff_show/top10_flagged/*.png) -- no fixed threshold applied to everyone
# - Inconsistent on paper: YR08-mor (66.7%) and AH19-mor (47.4%) are in the same contamination range and are NOT excluded
# - Do not cite this list as an established rejection criterion
SUBJECTS_EXCLUDE = ["DA01", "ES29/eve", "MG14/mor", "MS13/eve", "NB03", "YL26"]

# ── Channel ────────────────────────────────────────────────────────────────────
AIRFLOW_CHANNEL = "Airflow"

# ── Session map ────────────────────────────────────────────────────────────────
SESSION_MAP = {
    "mor": {"label": "Morning", "csv_suffix": "_2"},
    "eve": {"label": "Evening", "csv_suffix": "_1"},
}

# ── Cache sampling rate [Layer 1 — requires FORCE_RELOAD] ─────────────────────
# Raw signal is anti-aliased (Kaiser window) and stored at this rate.
# 25 Hz gives a 12.5 Hz Nyquist, comfortably above the ~3 Hz respiratory content.
CACHE_SFREQ = 25.0   # Hz

# ── Bandpass filter [Layer 1 — requires FORCE_RELOAD] ─────────────────────────
# Applied once to the native-rate raw signal, before downsampling to CACHE_SFREQ.
# This is a WIDE anti-alias / DC-removal pass, not a respiration-isolation
# filter — it deliberately keeps cardiac/movement content so NK2's own
# cleaning step can isolate the respiration band downstream (see
# RSP_CLEAN_METHOD below). Must run before the resample: at CACHE_SFREQ=25 Hz
# the Nyquist (12.5 Hz) is below AIRFLOW_LOWPASS, so this cutoff can't be
# applied after downsampling.
AIRFLOW_HIGHPASS = 0.01   # Hz
AIRFLOW_LOWPASS  = 70.0   # Hz

# ── NeuroKit2 cleaning method [Layer 2 — free to change] ──────────────────────
# Must stay on. The khodadad2018 peak detector expects the CLEANED signal
# (~0.05–3 Hz respiration band), not the wideband Layer-1 trace above.
# Confirmed by testing: with cleaning off, the detector was fed 0.01–70 Hz
# data, found ~1 breath/min instead of ~12–19, and 91% of trials failed the
# baseline+response peak requirement (rejected as no_cycles_found). Turning
# cleaning back on dropped rejection to ~23%.
RSP_CLEAN_METHOD         = "khodadad2018"
RSP_PEAK_METHOD_CLEANING = "khodadad2018"

# ── Display window [Layer 2 — free to change] ─────────────────────────────────
# Trims each trial's epoch (variable length, from trial_epochs.py) to this
# window FOR PLOTTING ONLY. Scoring and rejection always use the full
# baseline (t<0) / response (t>=0) span, regardless of this setting.
ANAL_TMIN, ANAL_TMAX = -5.0, 86.0

# ── Scoring method ─────────────────────────────────────────────────────────────
# This is the ONLY place SCORING_METHOD is set. Do not reassign it elsewhere
# in this file — a later assignment would silently win and this one would be
# dead code with no error raised.
PERFORM_SCORING = True
SCORING_METHOD  = "glm_deconvolution"   # or "peak_excursion_normalized"

# ── Trial-type classification ─────────────────────────────────────────────────
# True  = use the subject's own rating (arousal >= 7 or valence <= 3)
# False = use the image's pre-assigned category (negative/neutral)
# Standing default across pipelines per user decision, 2026-07-02.
USE_SUBJECTIVE_TRIAL_TYPE = True


# ══════════════════════════════════════════════════════════════════════════════
# BxB (peak_excursion_normalized) — settings and rejection gates
# ══════════════════════════════════════════════════════════════════════════════
# Score = max absolute excursion from baseline in the response window,
# normalised by the session-median RSP_Amplitude (i.e. a "typical breath" for
# that subject/session). 1.0 = response as large as a typical breath; 2.0 =
# twice as large. Runs in airflow_amp_processor.py.

# Gate 1 — noisy_baseline: z-score of a trial's baseline std vs. the session's
# baseline-std distribution. Raised to 3.0 for BxB because single-peak scoring
# is less sensitive to window noise than a full-window fit.
AIRFLOW_Z_SCORE_THRESHOLD = 3.0   # None = disabled

# Gate 2 — amplitude spike gate: REMOVED. max|RSP_Clean| couldn't distinguish
# a genuinely deep breath from an artifact; outlier scores are now caught by
# the score_ceiling gate (Gate 5) instead.
AIRFLOW_AMPLITUDE_Z_THRESHOLD = None   # disabled — do not re-enable without a new gate design

# Gate 3 — flat_signal: reject if baseline std < ratio × session-median std.
AIRFLOW_MIN_STD_RATIO = 0.1   # None = disabled

# Gate 4 — rate_artifact: reject if NK2's RSP_Rate anywhere in the epoch
# exceeds this. Normal resting rate is 12–20 bpm, up to ~40 bpm under stress
# (40 bpm = 1.5 s/breath). Anything faster is a movement artifact or a cough
# misread as a breath.
RSP_RATE_ARTIFACT_THRESHOLD = 40   # bpm; None = disabled

# Gate 5 — score_ceiling: reject if the normalised score exceeds this.
# A score of 3 means "3x the size of a typical breath" — implausible as a
# genuine response, so treated as an artifact.
AIRFLOW_SCORE_MAX = 3.0   # None = disabled

# Gate 6 — flat_response: same idea as Gate 3, but on the response window
# (t > 0) instead of baseline. Catches breath-holding or sensor disconnects
# that happen AFTER the trigger.
AIRFLOW_POST_MIN_STD_RATIO = 0.1   # None = disabled

# Gate 7 — atypical_shape: build a (n_trials × n_samples) matrix of each
# trial's waveform over the shape window below, compute the session's mean
# waveform (the "centroid"), and reject any trial whose mean-squared distance
# from that centroid exceeds mean(MSD) + SD_THRESHOLD × std(MSD). Catches
# flatlines, breath-holding, and disconnects that slip past the baseline/
# response std gates above. Used identically by both the BxB and GLM paths.
AIRFLOW_SHAPE_REJECTION_ENABLE = True    # master on/off switch
AIRFLOW_SHAPE_SD_THRESHOLD     = 3.0     # cutoff = mean(MSD) + k × std(MSD)
AIRFLOW_SHAPE_WINDOW_MIN       = -5.0    # seconds re t=0, shape window start
AIRFLOW_SHAPE_WINDOW_MAX       = 10.0    # seconds re t=0, shape window end


# ══════════════════════════════════════════════════════════════════════════════
# GLM (glm_deconvolution) — settings and rejection gates
# ══════════════════════════════════════════════════════════════════════════════
# Alternative to the BxB peak-picking scorer above: fits a canonical-response-
# function GLM per trial over its own [baseline_range, response_range) window
# (unchanged from trial_epochs.py), instead of comparing two single peak
# samples. Runs in airflow_glm.py. Uses the same cached signal_raw as BxB
# (CACHE_SFREQ=25 Hz) — no MFF reload needed to switch to this path.

# --- Phase 1: mean-center + respiration-band filter + downsample + z-score ---
# (on cached signal_raw). GLM_CYCLE_BANDPASS below is applied HERE, at the
# native 25 Hz rate, BEFORE downsampling to GLM_TARGET_SFREQ -- matching
# pspm_resp_pp.m Stage 1 exactly (references/pspm-preprocessing.md): the SAME
# filter that isolates the respiration band also serves as the anti-alias
# filter for the downsample (0.6 Hz sits 8.3x below the 5 Hz Nyquist at
# target_sfreq=10, ample margin) -- no separate anti-alias-only filter is
# needed for a signal this narrowband. There used to be an independent
# GLM_ANTIALIAS_LP=5 Hz stage here; removed since it duplicated (and gave
# less margin than) what the respiration-band filter already provides.
# --- Phase 0: despike signal_raw BEFORE any filtering (see despike_signal_raw
# in airflow_glm.py) ----------------------------------------------------------
# The 0.01 Hz highpass below is a zero-phase filtfilt whose impulse response
# rings for hundreds of seconds, so a single native-sample sensor glitch
# (electrode pop / motion) turns into a huge, session-dominating excursion in
# raw_z far outside its own brief duration -- e.g. RP06 eve: one 0.12s glitch
# at t=503.4s (540 session-robust-SD in signal_raw) inflated raw_z to -86 SD
# from t=345-946s. GLM_DESPIKE_K=50 is well above the largest genuine breath
# seen across every subject/session in this dataset (~35 session-robust-SD in
# signal_raw) and well below actual glitches (RP06 eve: 540/520/88).
# GLM_DESPIKE_MAX_RUN_SEC caps the flagged run length so a SUSTAINED large
# excursion (whole-session contamination, e.g. DA01) is left untouched --
# that's a SUBJECTS_EXCLUDE decision above, not a despike target.
GLM_DESPIKE_ENABLED     = True
GLM_DESPIKE_K           = 50.0   # median + k*MAD (native signal_raw units) gate
GLM_DESPIKE_MAX_RUN_SEC = 1.0    # flagged runs longer than this are left alone

GLM_TARGET_SFREQ = 10.0    # Hz, downsample target (25 -> 10 Hz is an exact 2/5 ratio)
GLM_ZSCORE_RAW   = True    # z-score the filtered/downsampled trace, session-wide.
# Z-scoring is a linear transform — it doesn't move zero-crossings, so cycle
# detection and RP timing are unaffected, but it gives a standardized scale
# for the rejection gates below. Note: RA/RFR are NOT reported in SD units —
# they're measured on signal_raw (the pre-z-score native cache), matching
# PsPM's resp-vs-newresp separation (see references/pspm-preprocessing.md in
# the pspm-respiration-audit skill).

# --- Phase 1 filter cutoffs / Phase 2 cycle detection (median + zero-crossing
# only, on the already-filtered Phase-1 trace) -------------------------------
GLM_CYCLE_BANDPASS = (0.01, 0.6)   # Hz, TWO cascaded 1st-order Butterworth filters
                                    # (lowpass 0.6 Hz, then highpass 0.01 Hz), each
                                    # bidirectional (filtfilt) -- matches pspm_resp_pp.m
                                    # Stage 1 exactly (lporder=1, hporder=1, direction='bi'),
                                    # not a single combined 2nd-order bandpass design.
                                    # Applied in Phase 1 (phase1_filter_downsample),
                                    # before downsampling -- see note above.
GLM_MEDIAN_WIN_SEC = 1.0   # seconds, median filter applied before zero-crossing detection
GLM_REFRACTORY_SEC = 1.0   # seconds, minimum spacing between accepted cycle onsets

# --- Phase 3: per-cycle RP/RA/RFR -> continuous series -----------------------
# "Sensitivity filter" high-pass, PER METRIC. PsPM (pspm_init.m defaults.glm)
# does NOT share one high-pass across the three respiration channels: the evoked
# (_e) modelspecs give RP a 0.01 Hz high-pass but RA and RFR 0.001 Hz
# (rp_e: hpfreq=0.01; ra_e/rfr_e: hpfreq=0.001; all hporder=1, direction uni).
# RP is a timing (period) series, so PsPM strips more slow drift from it; RA/RFR
# keep more low-frequency response energy at 0.001 Hz. One shared cutoff would
# collapse this documented distinction -- here it only mis-filtered RP, since
# RA/RFR were already at 0.001. build_continuous_series still accepts a plain
# scalar (applied to all three) for backward compatibility, but this per-metric
# dict is the PsPM-faithful default.
GLM_FINAL_HP = {"RP": 0.01, "RA": 0.001, "RFR": 0.001}   # Hz, per-metric high-pass
GLM_FINAL_LP = 1.0     # Hz, "sensitivity filter" low-pass (shared; PsPM lpfreq=1 for all)
# Both applied UNIDIRECTIONALLY (lfilter, not filtfilt) — the source method
# specifically calls for a unidirectional (causal) filter here.

# --- Phase 4: canonical response functions -----------------------------------
# h(t) = exp(-(t-tau)^2 / (2*sigma^2)) — a Gaussian bump centered at tau with
# width sigma, one per metric, describing the expected shape of a startle
# response in that metric.
GLM_RF_PARAMS = {
    "RP":  (4.20, 1.65),   # (tau, sigma) seconds — expected decrease/deceleration
    "RA":  (8.07, 3.74),   # expected increase / deeper breath
    "RFR": (6.00, 3.23),   # expected increase in flow
}
GLM_USE_DERIVATIVE = {"RP": False, "RA": True, "RFR": True}   # add dRF/dt as an extra regressor

# --- GLM rejection gates ------------------------------------------------------
# Same names/thresholds/intent as the BxB gates above, computed on the
# z-scored Phase-1 trace (raw_z) and the cycle-derived rate series instead of
# NK2 features — kept at parity so switching to GLM doesn't silently accept
# noisier trials than BxB does.
GLM_Z_SCORE_THRESHOLD       = 3.0    # noisy_baseline; mirrors AIRFLOW_Z_SCORE_THRESHOLD
GLM_MIN_STD_RATIO           = 0.1    # flat_signal;    mirrors AIRFLOW_MIN_STD_RATIO
GLM_POST_MIN_STD_RATIO      = 0.1    # flat_response;  mirrors AIRFLOW_POST_MIN_STD_RATIO
GLM_RATE_ARTIFACT_THRESHOLD = 40     # bpm, rate_artifact; mirrors RSP_RATE_ARTIFACT_THRESHOLD

"""
cycle_gap gate — why it exists:

A trial can have one real cycle in its baseline AND one real cycle in its
response window (so it passes both the custom detector and the NK2
cross-check below) while still containing a long DEAD stretch in between.
Confirmed by inspection in two real trials: AB22-eve T28 (one cycle at
t=-4.7s, nothing again until t=+11.8s — a 16.5s gap) and AH19-mor T13
(15.5s gap). Both are ~10+ seconds of genuinely flat signal.

Neither of the existing gates catches this:
  - flat_signal / flat_response look at std over the WHOLE baseline/response
    span — a couple of real cycles sitting at the edges keeps that std
    looking normal.
  - no_cycles_found doesn't fire either, since both detectors DO find
    something — just not continuously.

Fix: an ABSOLUTE minimum-rate floor (not a session-relative ratio — an
earlier attempt at a relative "weak_cycles" threshold broke on EV15-eve).
No adult breathes slower than ~5 bpm (12s/cycle) while awake, so a gap
longer than that — anywhere in the trial's own window, including from the
window edge to the first/last cycle — means detection genuinely lapsed,
not that this one breath was unusually slow. Same floor for every subject.
"""
GLM_MIN_RATE_THRESHOLD = 5   # bpm; longest gap between consecutive cycle onsets
                             # (or window edge -> first/last onset) must imply
                             # at least this rate, or the trial is rejected as cycle_gap

"""
no_cycles_found gate — why it's an NK2 cross-check, not a custom threshold:

The custom Phase-2 zero-crossing detector (mean-center -> bandpass -> medfilt
-> zero-crossing) has no amplitude/shape floor: it will count noise wiggles,
or a flat-baseline-then-step artifact, as valid "cycles" as long as they
cross zero on schedule.

First attempt: compare each trial's cycles to the session's own median RA/RP
(a ratio threshold). This fixed the known AH19-mor cases, but over-rejected
on EV15-eve (48.6% of that session — including trials that were visibly real
shallow breathing). A threshold fit to one session's amplitude spread doesn't
transfer to a session with a different spread.

Current approach: reuse NeuroKit2's own khodadad2018 detector (same
RSP_CLEAN_METHOD/RSP_PEAK_METHOD_CLEANING config the BxB path already applies
identically to every subject) as a cross-check — no new threshold to pick. A
trial is rejected if NK2 finds no real pre-onset peak AND post-onset
trough-then-peak in its window, regardless of what the custom detector
found. This exactly mirrors BxB's own has_cycles logic. It only affects
rejection — RP/RA/RFR themselves are still computed by the custom Phase 1-4
pipeline, unchanged.

Known remaining gap: this catches "no real breath structure" (flat traces,
step artifacts) but NOT amplitude-spike artifacts (a technically valid
peak-then-trough shape at an absurd amplitude, e.g. a cough or gross
movement). That's a separate failure mode, not addressed here.
"""

# score_ceiling — kept separate from AIRFLOW_SCORE_MAX (the BxB ceiling):
# GLM scores are regression coefficients, not "x times typical breath
# amplitude", so the two aren't on the same scale. score_rp/score_rfr are on
# the RP/RFR series' own scale (seconds / seconds-per-second); score_ra is in
# native signal_raw units (NOT SD units, since RA is measured pre-z-score —
# see GLM_ZSCORE_RAW above). So even the two RA-like ceilings aren't
# comparable to each other.
GLM_SCORE_MAX = None   # None = disabled until real beta distributions have been seen


# ── GLM artifact-handling method [Layer 2 — free to change] ───────────────────
# The ONE knob for the amplitude-spike gap the rejection gates above deliberately
# do NOT cover: a cough / gross movement / sensor swing read as a single giant
# "breath". (The no_cycles_found note above spells out why those gates catch
# "no real breath" but not "a real-SHAPED breath at an absurd amplitude".)
#
# This setting is layered ON TOP OF, not instead of, SUBJECTS_EXCLUDE above --
# that list always parks its hand-flagged whole-session contamination first (a
# check the Hampel bound structurally can't do, since it's relative to each
# session's own median/MAD). GLM_ARTIFACT_METHOD then decides how every
# remaining session's individual cycles/trials get handled. All three methods
# are a-priori and symmetric across Evening/Morning -- none ever looks at the
# Eve-vs-Mor result, so none can manufacture the hypothesised direction. Pick ONE:
#
#   "manual_exclude"
#       No further per-cycle action beyond the SUBJECTS_EXCLUDE skip that always
#       runs. Coarsest -- relies entirely on hand inspection, not a statistical
#       threshold.
#
#   "hampel_drop_cycles"
#       Finest-grained. Within EACH surviving session, flag any breathing cycle
#       whose RA exceeds a Hampel UPPER bound  median(RA) + GLM_ARTIFACT_K * MAD(RA)
#       (MAD scaled by 1.4826). Upper bound only, so it targets spikes and never
#       rejects shallow real breaths -- unlike the earlier session-median *ratio*
#       attempt that over-rejected EV15-eve. Flagged cycles are DROPPED before
#       the Phase-3 series is interpolated, so the spike never reaches any RA/RFR
#       series or GLM fit; the existing cycle_gap gate then rejects any trial
#       left with too long a dead stretch.
#
#   "hampel_reject_trials"
#       Middle granularity. Same Hampel flag, but instead of editing the series,
#       any trial whose analysis window [baseline_start, response_end) contains a
#       flagged cycle onset is rejected wholesale (reason "amplitude_artifact").
#       Keeps the series intact; discards more usable data per artifact than
#       dropping the single cycle -- and does so in an all-or-nothing way: ONE
#       moderately-deep real exhale near the edge of an otherwise pristine
#       15-20s window throws out the entire trial's GLM fit. Confirmed on real
#       data (2026-07-13): EE10-eve trials 4/5/7/8/22/23/25/27 and LG07-eve
#       trials 13/16/22/32/36 were all being rejected this way for a single
#       tail-end cycle (RA 7-10 MAD-units above session median) sitting inside
#       an otherwise clean, regular breathing trace -- confirmed by eye against
#       the raw trial trace, not a sensor glitch. Cranking GLM_ARTIFACT_K up
#       per newly-found case is whack-a-mole: the pooled cross-subject RA
#       MAD-multiple distribution has no clean gap (it decays smoothly out
#       past 20), so there is no universal "safe" k for this consequence.
#
# RESIDUAL LIMITATION shared by both hampel_* methods: the bound is RELATIVE to
# each session's own median/MAD, so on its own it would catch ISOLATED spikes
# inside an otherwise-normal session but miss a uniformly-corrupted session
# (e.g. DA01's session-wide saturation -- every cycle huge, median huge too).
# That's exactly why SUBJECTS_EXCLUDE is no longer conditional on this setting:
# it now covers the whole-session failure mode unconditionally, so hampel_* can
# safely be the default for everything else.
GLM_ARTIFACT_METHOD = "hampel_drop_cycles"   # "manual_exclude" | "hampel_drop_cycles" | "hampel_reject_trials"
                            # DEFAULT as of 2026-07-13: switched from
                            # hampel_reject_trials because the whole-trial
                            # consequence was too destructive for a single
                            # deep-breath cycle (see limitation above) -- dropping
                            # just the flagged cycle and interpolating over it
                            # keeps the rest of a real, clean trial's data instead
                            # of discarding it. The existing cycle_gap gate still
                            # rejects a trial if the drop leaves too long a dead
                            # stretch, so a genuinely artifact-heavy trial is still
                            # caught, just via a different rejection_reason.
GLM_ARTIFACT_K      = 3.5   # Hampel k in median + k*MAD. NOT the standard constant --
                            # the commonly-cited default is k=3 (Hampel 1974; Leys et al.
                            # 2013 recommend b=3 conservative / b=2.5 moderate). Reset to
                            # this looser-than-standard value (from an interim 7.0 tried
                            # under hampel_reject_trials) now that the consequence of a
                            # false-positive flag is a single-cycle interpolation rather
                            # than losing the whole trial -- a lower/stricter k is fine to
                            # live with under the gentler method. Verified on the cache
                            # (2026-07-13): at k=3.5 with hampel_drop_cycles, all of
                            # EE10-eve's and LG07-eve's previously-misflagged trials above
                            # pass, and total pipeline-wide rejected trials fall from 204
                            # to 162 (net fewer, not shifted elsewhere) vs the prior
                            # hampel_reject_trials/k=7.0 setting.
                            # Only the hampel_* methods use it.


# ── GLM confirmatory endpoint [Layer 2 — free to change] ──────────────────────
# The SINGLE metric the headline Evening-vs-Morning test is run on (in
# airflow_glm_show.ipynb). Naming it a priori is the multiplicity discipline that
# keeps the reported p honest: the notebook computes many Wilcoxons (RP/RA/RFR x
# valence x ratios), and quoting whichever came out smallest would be tailoring
# to the data. RP/RFR and the valence contrasts stay shown as SECONDARY /
# exploratory.
#
#   "RA"  (DEFAULT, recommended) -- respiration amplitude: the most direct
#         measure of evoked breath SIZE / reactivity, and the metric PsPM equips
#         with a derivative regressor (its richest evoked model). Cleanest map to
#         "startle response magnitude".
#   "RFR" -- respiratory flow rate (RA/RP): depth and speed together; flow-native
#         for an airflow transducer.
#   "RP"  -- respiration period (timing/deceleration); PsPM treats it differently
#         (no derivative, tighter high-pass) -- least direct "reactivity" read.
#   "composite" -- mean of the three z-scored per-subject condition means; robust
#         if no single metric dominates, but not tied to one PsPM basis function.
GLM_PRIMARY_METRIC = "RA"   # "RA" | "RFR" | "RP" | "composite"


# ── GLM estimation architecture [Layer 2 — free to change] ────────────────────
# HOW the beta that becomes score_<metric> is estimated. Both paths use the SAME
# Phase 1-3 output (the continuous RP/RA/RFR series) and the SAME rejection gates,
# canonical response functions, orthogonalized derivative and per-modality
# filters — they differ ONLY in the regression that turns that series into a
# beta. See GLM_METHOD_FOUNDATIONS.md section 5 for the architectural contrast.
#
#   "per_trial"  (DEFAULT — original behaviour, results unchanged)
#       One independent regression per trial, scoped to that trial's own
#       [baseline_start, response_end) window (fit_trial_glm). Produces one beta
#       PER TRIAL. This is what enables per-trial diagnostics and the trial-level
#       scatter/timecourse plots. It is NOT what Bach 2016 / pspm_glm.m does; at
#       this SNR it is a low-power estimator (short windows, one solve per trial,
#       medians ~1e-4), and the confirmatory Eve>Mor test is null on it.
#
#   "pooled_session"  (PsPM pspm_glm.m architecture)
#       ONE design matrix for the WHOLE session with one regressor set per
#       CONDITION (valence label 1=Neg / 2=Neu among that session's ACCEPTED
#       trials): each condition's event train (a unit impulse at every accepted
#       trial's onset) is convolved with the canonical basis (+ orthogonalized
#       derivative for RA/RFR), the convolved columns are mean-centred
#       (PsPM model.centering=1) and a SINGLE least-squares fit is solved ->
#       ONE beta per condition. Every accepted trial then receives its
#       (session, condition) beta as score_<metric>; rejected trials keep NaN.
#       This is "similar to standard analysis of fMRI data" (pspm_glm.m) and is
#       a strictly more powerful estimator — one solve over ~all a condition's
#       trials instead of averaging ~17 near-zero single-trial betas. The
#       confirmatory cell's per-subject-per-session mean then collapses to a
#       trial-count-weighted mean of the Neg/Neu betas, Evening vs Morning.
#
# The direction (Eve>Mor) and primary metric (GLM_PRIMARY_METRIC) are fixed a
# priori and are NOT chosen from the estimation path's result — switching path is
# a pre-registered sensitivity/power change, not p-hacking (do not flip back and
# forth watching the p; pick the reported path a priori).
GLM_ESTIMATION = "per_trial"   # "per_trial" | "pooled_session"

# ── External data ──────────────────────────────────────────────────────────────
SUBJECTS_XLSX = r"/Volumes/My Passport/startle_raw/subjects.xlsx"

# ── Cache ──────────────────────────────────────────────────────────────────────
FORCE_RELOAD = False   # cache rebuilt 2026-07-05 (find_mff_file fallback + AppleDouble fix applied).

# ── Output ─────────────────────────────────────────────────────────────────────
VERBOSE  = True
PLOT_DPI = 150

# ── Colours ────────────────────────────────────────────────────────────────────
NEG_COLOR  = "#C0392B"
NEU_COLOR  = "#2980B9"
REJ_COLOR  = "#00000000"
EVE_COLOR  = "#E67E22"
MOR_COLOR  = "#8E44AD"
UP_COLOR   = "#27AE60"
DOWN_COLOR = "#E74C3C"
