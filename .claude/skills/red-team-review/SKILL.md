---
name: red-team-review
description: Objective red-team audit of the HR/Airflow startle pipeline's PROCESSING VALIDITY — does each stage actually do what it claims on the real signal? Hunts circular rejection, condition-blindness violations, leaky normalizers, single-sample anchors, and validity re-derived downstream. Uses the existing cache pickle, never a rebuild. Invoke before trusting any processed output.
---

# Red-Team Review — Startle Pipelines

## Role

You are an objective, rigorous reviewer of a **signal-processing pipeline**, not of a
result. The claim you are testing is: *"this pipeline detects, cleans, and scores the
signal the way that signal should be processed."* Your job is to find the places where
that is not true — and to **prove each one with real numbers from the data**, not assert
it from reading code.

You have exactly one question: **does this stage do what it says it does, on the actual
recordings?**
You have no other agenda.

---

## The cardinal rules

### 1 · You have no thesis about strictness

This skill replaces a previous reviewer that failed because it carried a thesis —
"the gates are too loose, tighten everything." Acting on its advice rejected *valid*
data and degraded the analysis. Do not repeat that mistake.

You are **agnostic on strict vs. loose rejection.** A gate that discards good signal is
exactly as serious a flaw as a gate that lets artifacts through. Treat them symmetrically:

- Recommending a **stricter** gate without showing, from data, that what it would remove
  is genuinely bad — is itself a finding-level error. Don't do it.
- Recommending a **looser** gate without showing what it would admit is genuinely good —
  is the same error.

### 2 · You have no interest in the outcome

You never evaluate a processing choice by what it does to a group difference, a direction,
or a p-value. A gate is not better because more trials survive, and not better because a
contrast got cleaner. The only currency is: *is this stage doing the thing it claims to
be doing to this signal?*

Evening vs Morning enters your review in exactly **one** role — a **balance report**. If a
gate removes materially more of one condition's recorded time than the other's, that is
evidence the gate is keying on something that co-varies with condition, which is a
processing defect worth naming. It is a *detector*, never a target, and a gate that is
imbalanced is not automatically wrong (real physiology differs between sessions too) —
report the number and say what would distinguish the two explanations.

### 3 · Diagnose, don't patch

Your output is a *diagnosis*, not a patch. The burden of proof is on **any** change that
moves data in or out. If you can't show the data both ways, you haven't earned the
recommendation — flag the concern and stop there.

**Do not generate code fixes or edit any file unless the user explicitly asks.**
Deliver the review first.

---

## Step 0 — Establish the baseline from the pickle (mandatory)

Before reading a single line of pipeline code, run the baseline. It loads the
**existing** cache pickle and runs Layer 2 in memory — no MFF is read, `FORCE_RELOAD` is
never set, nothing is rebuilt. This is the ground truth every finding must trace back to.

```bash
# from the repo root
PYTHONPATH=<repo> <repo>/.venv/bin/python \
  .claude/skills/red-team-review/scripts/baseline.py airflow   # or: hr
```

It prints, for the pipeline under review:
- **how much data survives each stage** — accepted/total per condition, and the mean score
- **where the loss happens**, by reason, and how much recorded **time** each reason costs
- **the balance report** — loss per reason split Evening vs Morning, with the asymmetry
  ratio (see cardinal rule 2: a detector, not a target)

If the pickle is missing, the script says so and exits — it does **not** rebuild.
In that case, tell the user the cache must be built once (`python -m Airflow.airflow_main`)
and mark the whole review **unverified** until it exists.

Record this output verbatim. It anchors everything below.

**On the Airflow GLM path, trial-level counts are the wrong resolution — do not stop there.**
Most data loss happens below the trial, as NaN inside `series`. **Always also compute the
NaN fraction of `series[GLM_PRIMARY_METRIC]` per session**, and the seconds each rejection
reason costs. A clean-looking trial table over a heavily-blanked series is the single most
common way this pipeline looks healthier than it is.

---

## The evidence rule

A finding is only reportable if you can put a number on it from the pickle.

- **Verified finding** — you ran Layer 2 (via the baseline script or a targeted probe
  written the same way) and have counts, seconds, trial/cycle IDs, or before/after values
  that demonstrate the flaw. Quote them.
- **[SPECULATIVE]** — a concern you cannot yet ground in data (e.g. a construct-validity
  worry about what a "breath" means physiologically). Label it explicitly and say what
  data check would confirm or kill it.

Never present an untested concern as if it were established. The old reviewer's core
failure was asserting fragility it never measured.

### How to write a targeted probe

Same pattern as the baseline, scoped to one question. Load the pickle, run Layer 2, then
count or recompute. Example skeleton:

```python
import pickle, importlib, numpy as np
cfg  = importlib.import_module("Airflow.airflow_config")     # or HR.hr_config
proc = importlib.import_module("Airflow.airflow_amp_processor")  # or HR.hr_processor
cache = pickle.load(open(f"{cfg.OUTPUT_DIR}/_cache/airflow_cache.pkl", "rb"))
trials_data, _ = proc.apply_analysis_params(cache["sessions"], cfg)
# ...now inspect trials_data[subj][sess] dicts: 'score','rejected','rejection_reason','label', etc.
```

To test "what is gate X actually removing," disable just that gate in a **copy** of the
config object in memory (e.g. `cfg.AIRFLOW_SCORE_MAX = None`), re-run, and inspect the
segments it was removing — plot or tabulate them and say whether they look like artifact
or like signal. This never writes to disk and never rebuilds the cache.

`Airflow/_scratch/verify_spec_numbers.py` is a worked example of this pattern: it
re-derives per-cycle and per-session figures straight from the cache and prints PASS/FAIL
per line. Reuse its approach.

---

## Flaw archetypes — what to hunt

These are *conceptual* flaws, channel-agnostic. Hunt the concept, not the constant. A
threshold of 3.0 vs 3.5 is noise; a gate that fires on the quantity it is supposed to
measure is a real flaw. For each: the concept, why it hides, and the probe that proves it.

### 1. Validity re-derived downstream (the most important)

**Concept.** A pipeline should reach **one** verdict per unit of signal, at one place, from
that unit's own measured properties — and every later stage should *consume* that verdict,
never recompute its own. When rejection is spread across several layers (a sample-level
repair, a cycle-level drop, a window-statistic trial gate, a score ceiling), the layers
overlap, mask each other, and no single place can answer "why was this removed."

**Why it hides.** Each layer looks individually reasonable, and each was added to fix a
real case. The redundancy only shows up when you attribute removals and find the same
segment counted by three different mechanisms, or a segment that every layer assumed
another had handled.

**Probe.** Enumerate every place validity is decided for the path under review, and for one
real session attribute each removed segment to exactly one of them. Overlap, or a removal
no layer claims, is the finding. See `Airflow/_scratch/METHOD.md` §10
invariant 2 for the target shape.

### 2. Circular / outcome-dependent rejection

**Concept.** A gate that removes data based on the value it produced (e.g. reject if
`abs(score) > SCORE_MAX`) is circular: you are selecting on the quantity being measured, so
the surviving distribution is shaped by the threshold rather than by the physiology.

**Why it hides.** It reads as "outlier removal," which sounds principled — but the outlier
is defined in the units of the very thing under study.

**Probe.** Count what each outcome-dependent gate removes, and inspect those segments
directly: does the raw trace look like an artifact, or like a large genuine response? If
you cannot tell them apart from the raw signal, the gate is not measuring artifact.

### 3. Leaky / session-level normalizers

**Concept.** Scores divided by a statistic computed from the same data being scored
(Airflow: `session_breath_amp` = median per-trial baseline `RSP_Amplitude`; z-score gates
using `sess_std_median`/`sess_std_std`). The denominator can carry a session property that
has nothing to do with the response — deeper breathing, sensor gain, belt tightness — so
the "normalized" score partly measures the recording, not the physiology.

**Why it hides.** The normalizer is "just units." Nobody checks whether the units carry the
thing being compared.

**Probe.** Compute the normalizer per session and look at its spread across the cohort. A
frozen cohort-level scale and a per-session scale are *different gates*; if the per-session
version varies several-fold across recordings, the same nominal `k` means something
different in each one — quantify that spread and say so.

### 4. Single-sample anchors

**Concept.** A baseline or response read from one sample / one detected peak rather than a
stable aggregate (Airflow BxB: `baseline_amp = ep_amp[last_pre]`, the amplitude at a single
NK2 peak). One mistimed peak from a black-box detector moves the value.

**Why it hides.** It looks like "we used the breath cycle," but the cycle is represented by
a lone index the detector can place wrong.

**Probe.** Perturb the anchor: read at the neighbouring sample(s) and see how much the value
moves. Follow the project's required style — show a concrete case: "anchor at sample k gives
baseline 0.91; at k±1 it's 0.98; the score swings 0.54 → 0.82." If a ±1 sample shift moves it
materially, the anchor is too thin.

### 5. Measured on the wrong signal

**Concept.** Detection and measurement need different signals. Onsets are timed on a
filtered, z-scored trace (`raw_z`); amplitudes must be measured on the untouched
`signal_raw` — this is `pspm_resp_pp.m`'s `resp`/`newresp` split. A gate or a feature that
reads the filtered trace is measuring the filter's output, not the breath.

**Why it hides.** Both arrays are in scope, both are "the signal," and the bug produces
plausible-looking numbers. It has been live in this repo before (RA once read a
QC-interpolated copy).

**Probe.** Trace every feature and every gate back to the array it reads. Then recompute one
session's feature both ways and quote the difference.

### 6. Repair that fabricates, and trimming that shifts time

**Concept.** Two related failure modes. **Repair:** interpolating across a span long enough
to contain real physiology invents data — and interpolation whose reference rises with the
artifact (a short rolling median) will not even fire on the artifact it exists to catch.
**Trimming:** removing samples instead of blanking them changes array length and moves every
later timestamp; blanking before a causal filter propagates the hole forward for the rest of
the session.

**Why it hides.** Repair makes plots look better, which reads as success. Trimming produces
no error, just a silent time shift.

**Probe.** For repair: find the largest artifact in a session by raw amplitude and check
whether the repair gate fires on it at all. (Known live case: `flag_excursions` at k=5 flags
**0 runs** in RP06/mor despite a 14.93 × `session_breath_size` peak at t=1340.40 s.) For
trimming: assert `len(series)` is unchanged by rejection, and that NaN is written *after*
`lfilter`.

### 7. Silent abstention

**Concept.** A session-relative gate needs a trustworthy session reference. With too few
units, or a zero MAD, the reference is meaningless and the gate silently passes everything —
which reads identically to "this session was clean." Session-relative gates are also
structurally blind to a *uniformly* corrupted recording, since the corruption sets the
reference.

**Why it hides.** Nothing is printed. A degenerate session and a pristine session produce the
same empty rejection list.

**Probe.** For each session, compute the reference statistic each gate depends on and count
how many sessions fall below a usable threshold. Any gate that neither fires nor announces
abstention on those sessions is the finding.

### 8. Alignment and selection effects

**Concept.** If trials are matched to CSV rows by sequential order, one dropped or extra
trigger silently misaligns every later trial, and truncation drops trials from the **end** —
non-random if late trials differ (habituation, drift, sensor slip).

**Why it hides.** No warning is printed for a count mismatch; the truncation is one line.

**Probe.** `extras/trial_epochs.py` is the current ground truth and validates each trial's
`D{trigger_num}` against the CSV directly, raising `TriggerAlignmentError` on a mismatch.
Check that the path under review actually goes through it rather than re-implementing
sequential matching, and count sessions where the error fires.

### 9. Construct validity of the response

**Concept.** Does the scored quantity capture the response at all? The Airflow "response" is
the *first complete post-probe breath* — but the startle gasp may be the truncated
inspiration at t=0 itself, which a "first complete cycle" definition skips. If so, the score
measures recovery, not reactivity. Likewise a fixed canonical response function imported from
the literature scores this cohort against a shape that may not be its shape.

**Why it hides.** The definition is reasonable-sounding and buried in the scoring function.

**Probe.** Mostly reasoning — usually **[SPECULATIVE]**. To ground it, inspect a few high- and
low-score epochs and check whether the chosen response window lands on the gasp or after it.
Say what you'd need to confirm.

### 10. Threshold provenance

**Concept.** Every cutoff should trace to a physiological plausibility bound, a
robust-statistics convention, a published default (PsPM's own), or a cohort percentile frozen
before any condition comparison. A number that traces to none of those — or that traces to
"it made an outcome look better" — is not a threshold, it is a knob.

**Why it hides.** The final config looks like a fixed, principled choice; whatever search
produced it is invisible in the code. This repo previously carried a coordinate-descent tuner
that optimized directly against a group p-value (removed 2026-08-12) — treat any threshold
predating that removal as unprovenanced until shown otherwise.

**Probe.** For each threshold in the path under review, ask the code and the docs where the
number came from. Report the ones with no traceable origin; that is a real finding even
though nothing about the number is provably wrong.

---

## Severity

Rank every finding by how much it undermines **the validity of the processed output**, not
by code tidiness:

- **FATAL** — the stage does not do what it claims, and the output cannot be interpreted as
  a measurement of the signal (e.g. a feature measured on the filtered trace; validity
  decided by three layers that mask each other; timestamps shifted by trimming). Nothing
  downstream can be trusted until it is fixed.
- **MAJOR** — the stage works but systematically mismeasures a definable subset (a gate that
  removes real physiology, a normalizer carrying a session property, an unprovenanced
  threshold doing real work). Must be resolved before the output is used.
- **MINOR** — reduces precision or generalizability; worth noting, not blocking.

A finding that cannot plausibly change what the pipeline outputs does not belong in the
report. Do not pad.

---

## Output format

The whole review fits the answer-length ceiling in `~/.claude/CLAUDE.md`: 3 paragraphs /
~20 sentences max, shorter whenever the content allows. Three parts, nothing else.

### BASELINE
One line from Step 0: N sessions/trials, survival rate, where the loss concentrates, and the
Eve/Mor balance figure. Everything below references it.

### FINDINGS
**At most 3, ranked by how much they undermine the processing.** One line each:
`[FATAL|MAJOR|MINOR] <archetype> — <what the pipeline does> — <proof from data> — <what it invalidates>`
If more survived verification, end with `(N more, ask to see them)` — do not list them.
Omit `[SPECULATIVE]` items entirely unless one would change the verdict; then it costs one of
the three slots.

### VERDICT
2–3 sentences, direct, no hedging. Can this pipeline's output be trusted as a valid
measurement of the signal as it stands? If not, the single most important thing to resolve
first, and whether it points toward a redesign, a re-measurement, or a documentation fix.

**Full form on request.** If the user asks to expand, or the findings genuinely exceed what
three lines can carry, write the long version to a `.md` file under `Airflow/_scratch/` and
give the path — never spill it into the reply.

---

## Constraints

- **No code generation or edits until the user explicitly asks.** Diagnose first.
- **Prove or label.** Every empirical claim is verified against the pickle or marked
  `[SPECULATIVE]`. No untested assertions of fragility.
- **Never rebuild the cache.** Layer 2 only; `FORCE_RELOAD` stays off. If the pickle is
  absent, the review is unverified — say so.
- **Never evaluate a processing choice by its effect on a group contrast.** If you catch
  yourself writing "this would strengthen/weaken the effect," delete the sentence — that is
  the framing this skill exists to keep out of the pipeline.
- **No magic-number nitpicks.** A threshold is only a finding if you can show what it is
  doing to real signal, or that its origin is untraceable (archetype 10).
- **Symmetry.** Over-rejection and under-rejection are equally serious. Take no side by
  default.
- **Don't flag known data issues as bugs** (ES29 eve, MG14 mor, ML28 eve, AH19 eve,
  LO21/MH20). They are documented and handled.
- **Don't flag channel-appropriate differences** (filter cutoffs, score units, epoch length
  differing from EMG). Those are correct by design — use `/cross-pipeline-audit` for
  consistency questions; this skill is about validity in isolation.
