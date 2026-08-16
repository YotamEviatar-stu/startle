# Method review — Airflow cycle-based rejection

Start a **fresh session at the repo root** (`/Users/yotameviatar/startle-1`), put it in **plan
mode** (shift+tab twice), and send exactly:

> Read `Airflow/_scratch/REVIEW_BRIEF.md` and follow it exactly.

---

## 0 · What kind of review this is

This is a **conceptual / methodological review, not a code review.** Do not audit style, naming,
error handling, or line-level correctness. The question is whether the *method* is sound: whether
"invalid breath" is defined on principled grounds, whether the resulting β₁ still measures what it
claims to measure, and whether the endpoint that comes out is defensible to a sceptical reviewer of
the paper.

Read `Airflow/_scratch/cycle_rejection_spec.md`, and read `Airflow/airflow_glm.py` /
`Airflow/airflow_qc.py` alongside it — those two are the only authoritative description of what
actually runs. `Airflow/CLAUDE.md`, `README.md`, `GLM_METHOD_FOUNDATIONS.md` and
`_scratch/cycle_pipeline_design.md` are stale; the spec's §0 says exactly how. The conceptual case
with figures is `_scratch/cycle_rejection_logic.html`.

Nothing is committed. You are reviewing a design before it is built.

## 1 · Protocol — interview first

Do **not** produce findings on this message.

1. Read the spec and the two `.py` files.
2. Ask a batch of **≤ 4 questions** with `AskUserQuestion` — only questions whose answer changes
   your verdict. **At most 2 rounds.** If you can answer it from the sources, don't ask it.
3. Restate in ≤ 10 lines: what you will review, what is out of scope, and what you will deliver.
4. Wait for an explicit **GO**. Then review.

## 2 · What the pipeline is trying to measure

Snore-sensor respiration → breath cycles → per-cycle validity verdict → RP/RA/RFR series carrying
NaN holes where breaths were rejected → CRF-GLM (PsPM / Bach et al. 2016) → β₁ per trial →
per-subject mean → Evening-vs-Morning paired Wilcoxon. `RA` is the single pre-registered
confirmatory endpoint.

We follow PsPM for cycle detection, filtering, the CRF basis and the GLM. We deliberately diverge by
being **more aggressive on artifacts**, because this sensor picks up coughs, subject movement and
sensor flinches as flat spans, extreme amplitudes and non-breath-shaped cycles. The motivating
number is in spec §6: in MS18/eve, four artifact breaths are 3.28 % of samples and carry **61.1 %**
of Σy². Both halves need judgement — is the aggression warranted, and does it cost anything the
paper would have to defend.

## 3 · The five questions this review must answer

Give each an explicit verdict: **sound** / **defensible with a stated limitation** / **not
defensible as designed**. Argue from the spec, the code and the real numbers in §6.

**Q1 — Is "invalid breath" defined by artifact properties, or by properties the response itself
has?** `extreme` rejects a cycle when `RA` exceeds `CYCLE_RA_MAX_RATIO` × session-median `RA` — and
`RA` is the endpoint. A startle response to a loud sound is plausibly a deep breath. Where is the
boundary between artifact and response drawn, is it drawn on evidence, and what happens to a real
large-amplitude response? Same question for `rate_implausible`'s upper bound (a sigh-with-pause) and
for `no_inspiration` vs a genuine post-startle apnoea.

**Q2 — Are the neighbour rules inferentially justified?** `recovery` rejects the cycle *after* an
`extreme` on the claim that it is a recovery slide, not a breath. `gap_fill` rejects cycle t when
t−1 and t+1 are both invalid. State what each rule asserts about the physiology, whether that claim
is supported, and what each costs when the artifact lands just before a stimulus — i.e. when the
removed neighbour is exactly where the response would be.

**Q3 — Is the missing-data model honest?** `fit_pooled_session_glm` solves on finite rows only
(`airflow_glm.py:687`), which treats the holes as ignorable. But rejection is amplitude-driven —
correlated with the signal being modelled — and the Eve-vs-Mor removal fractions are known to be
unbalanced. State what would have to be true for β₁ to stay unbiased, whether it is true here,
and in which direction the estimate moves if it is not. Consider that a hole removes rows from the
*response* portion of the window more often than from baseline.

**Q4 — Is β₁ still the quantity PsPM validated?** PsPM assumes a complete interpolated series.
Enumerate every divergence (holes in `y`, retiring `flag_deep_breaths`, dropping
`flag_excursions`' interpolation, the gate set, knot-to-knot bridging), say why each is warranted by
this sensor's noise, and flag any that silently changes the estimand rather than just cleaning it.
Use `/pspm-respiration-audit` for the fidelity half.

**Q5 — Does a defensible score come out the far end?** Not "is it non-NaN". Is there a minimum
surviving-data requirement before a session's β is trusted? Are βs estimated from very different
amounts of surviving data comparable in an unweighted paired test? Does a session-relative reference
(median `RA`, median `std`) make the gate *stricter in clean sessions and looser in dirty ones* —
and what does that do to a between-condition comparison? Resolve spec §5 (12 s admission window vs
~25 s trial window for blanking) on methodological grounds.

## 4 · Skills

Invoke `/startle-experiment` first, before reading anything else — it is the experiment briefing
(design, data layout, DIN triggers, EMG as the canonical reference) and this review needs it.
Then `/pspm-respiration-audit` for Q4, `/red-team-review` for Q3 and the circularity half of Q1,
and `/startle-research` if a claim needs grounding in the literature (respiratory startle response
morphology, apnoea criteria, artifact handling in respiration psychophysiology). Use
`/cross-pipeline-audit` only if you need the EMG reference's event logic. No MFF reloads — use
`airflow_output/_cache/airflow_cache.pkl` for any number you want to check.

## 5 · Out of scope / not up for debate

Reject per cycle and NaN its samples, never trim. Blank knot-to-knot. Features measured on
`signal_raw`, never `raw_z`. Condition-blind by construction; balance is *reported*, never used as
a gate, and is seconds-weighted. `SUBJECTS_EXCLUDE` is hand-verified and out of scope. Thresholds
must never be tuned against the Evening-vs-Morning endpoint — if you think a threshold is wrong,
argue it from physiology or from artifact evidence, never from what it does to the p-value. Judging
the design by whether it supports the hypothesis is itself a finding against you.

You may challenge anything above if you have a **mechanism**, not a preference.

## 6 · Output

After GO only. In chat: **at most 3 findings, one line each**, most severe first, then
`(N more, ask to see them)`. Full report to `Airflow/_scratch/METHOD_REVIEW_<YYYYMMDD>.md`, path in
the reply. Each finding carries: which of Q1–Q5 it belongs to, the mechanism that breaks, a concrete
numeric example from this cohort, and a verdict — *invalidates the endpoint* / *biases it in a
stated direction* / *carry as a limitation*. End the report with the one change that would most
improve credibility, and one sentence on what you could not check.
