Read these three files in order before doing anything, and no other Airflow doc until you have:

  1. Airflow/_scratch/IMPLEMENTATION_BRIEF.md   (rules, settled decisions, open ones, what to build)
  2. Airflow/_scratch/STATUS.md                 (status, statistics inventory, traps)
  3. Airflow/_scratch/cycle_rejection_spec.md   (design; §4 invariants, §6 scope warning, §8-10)

Then run both verifiers and confirm they are green before touching a line:

  .venv/bin/python Airflow/_scratch/verify_spec_numbers.py
  .venv/bin/python Airflow/_scratch/debug_pipeline.py

CONTEXT
The cycle-based rejection scheme is already implemented and clean — one verdict per breath,
five gates, admission by valid-breath fraction over the trial's own response epoch, one beta
per session. All legacy per-trial gates are deleted. This session builds the VISUAL INSPECTION
INSTRUMENT the method is accepted on. It does not change the method.

TASK — build ONE new notebook, Airflow/airflow_method_show.ipynb, with exactly two views.
Keep it lean: reuse compute_session_qc from airflow_qc_show.ipynb, add nothing else.

  View 1 — "main view", 3-layer pipeline, both of the subject's sessions side by side.
     Layer 1: signal_raw @ 25 Hz, rejected cycle spans shaded.
     Layer 2: raw_z @ 10 Hz with detected onsets.
     Layer 3: series[METRIC] with its NaN holes.
     X-axis is the ANALYSED WINDOW ONLY (union of trials' response_range_sec) — not the
     whole recording. Trial onsets marked, green admitted / red rejected.

  View 2 — "tagging view", cycle labelling, both sessions side by side.
     Paged 60 s rows over signal_raw across the analysed window. Every cycle boxed and
     coloured by its verdict; the gate name printed on each rejected cycle. Cycle ids are
     1-based within the analysed window.

  Nothing else. No statistics panel, no endpoint, no extra cells.

Then show me both on two subjects and stop for review.

HARD RULES (easy to violate — see brief §3)
  - One decision point per breath; nothing may re-derive validity downstream.
  - No function in the analysis path may read the session key, label, or has_sound.
  - The viewer is condition-blind too: label the two panels "Session A" / "Session B" in a
    fixed order and print eve/mor nowhere. Put the mapping behind REVEAL_SESSION_LABELS =
    False at the top of the notebook, off by default. The user wants to SEE both sessions,
    not to know which is which.
  - Never compute or consult the Evening-vs-Morning Wilcoxon. It is not the acceptance test.
  - Only 36.5% of detected cycles are inside the analysed window — scope and state every number.
  - Show raw, un-highpassed signal; decimate preserving peaks, never stride-subsample;
    keep sample counts low enough that MNE/matplotlib does not hang.
  - No comments or docstrings in source unless asked.
  - Change no threshold. Do not touch zscore_subject_series — it is parked.
  - Write no implementation code beyond the two views without asking first.

ANSWER LENGTH: 3 paragraphs max. Long-form goes to a .md file under Airflow/_scratch/ with
the path in the reply.
