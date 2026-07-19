---
name: startle-research
description: Domain-oriented literature/methodology research for the Startle experiment project (EMG/HR/Airflow startle-reflex analysis, PsPM toolbox conventions). Use when a question needs grounding in primary sources — a paper's method, a PsPM function's actual behavior, a physiological-signal-processing convention — rather than general research.
---

Spin up a **background agent** to do the research, so you keep working while it reads. This is the same mechanism as the generic `research` skill, pre-loaded with this project's domain so the agent doesn't start from zero.

Its job:

1. **Check `Startle/papers/` first.** This project already keeps a working literature set there (startle-reflex physiology, respiration/GLM methodology papers). Read what's already collected before searching externally — don't re-fetch something already on disk.
2. If the question concerns PsPM specifically (GLM basis functions, `pspm_resp_pp`, `pspm_glm`, cycle/breath detection), treat the actual PsPM source (bachlab/PsPM on GitHub) as the primary source — not a blog or secondary summary of it. The `/pspm-respiration-audit` skill already holds PsPM's source-level defaults in memory; check whether it already answers the question before dispatching new research.
3. Investigate against **primary sources** — official docs, source code, specs, papers themselves — following every claim back to the source that owns it, exactly as the generic `research` skill does.
4. Write findings to a Markdown file, citing each claim's source. Match this project's existing convention for method write-ups (e.g. `Startle/Airflow/GLM_METHOD_FOUNDATIONS.md`, `Startle/Airflow/DISCUSSION.md`) rather than inventing a new location.
5. State explicitly whether the finding changes anything in `Startle/emg_raw_potentiation.py` (canonical reference) or `Startle/trial_epochs.py` (epoch/trial extraction) — those are the files any methodology change would have to flow through.
