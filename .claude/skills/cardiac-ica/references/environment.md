# Running environment (this machine, verified 2026-09-30)

## Software

- MATLAB **R2026a** Update 4 at `/Applications/MATLAB_R2026a.app` (`MATLAB_BIN` in `hr/config.py`). Installed toolboxes: Signal Processing, Simulink, MATLAB MCP Server. **No Statistics and Machine Learning Toolbox.**
- FieldTrip at `~/Documents/MATLAB/fieldtrip` (`FIELDTRIP_DIR`), git `4c553bda8` (2026-07-23).
- SASICA at `~/code/tools/SASICA` (`SASICA_DIR`), master `9ab76d7`, with its bundled minimal EEGLAB in `SASICA/eeglab` (adminfunc, popfunc, sigprocfunc…, plugins `firfilt`, `ICLabel`).
- heart_functions: submodule `~/code/tools/SASICA/CARACAS/heart_functions` (the one `eeg_SASICA` adds to the path) and standalone `~/code/tools/heart_functions` (`HEART_FUNCTIONS_DIR`). Same `heart_peak_detect.m`.
- Shims: `~/code/tools/matlab_shim/{prctile,zscore,rep2struct}.m`; copies in `hr/matlab/`. `hr/caracas/caracas_session.m` instead copies `SASICA/private/zscore.m` into `hr/_cache/deps/` and adds that.

## Function resolution (after `restoredefaultpath; ft_defaults; addpath SASICA, CARACAS, heart_functions, SASICA/eeglab/functions`)

| Function | Resolves to | Needed by |
|---|---|---|
| `kurtosis`, `skewness`, `range`, `nanmean` | FieldTrip `external/stats` | heart_peak_detect (A, B), C |
| `prctile` | base MATLAB `toolbox/matlab/datafun` (shim shadows it; identical output, max diff 1e-16) | A, B, C |
| `zscore` | **not found** without shim | heart_peak_detect `nanzscore` → A, B |
| `rep2struct` | **not found** without shim | B |
| `chi2inv` | **not found** (Statistics Toolbox) | C → C cannot run here |
| `timepts`, `ifelse`, `mymkdir` | **not found** | B with `plot_heart_IC = 1` |
| `eeg_getdatact`, `eeg_emptyset`, `convertlocs` | `SASICA/eeglab/functions/...` | eeg_SASICA; building an EEG struct |

MATLAB private-folder rule: files in `SASICA/private/` are callable only from `SASICA/*.m` (e.g. `eeg_SASICA.m`), not from `SASICA/CARACAS/` or `heart_functions/`.

## Path recipe that works (as in `hr/caracas/caracas_session.m`)

```matlab
restoredefaultpath;
addpath(ftdir); ft_defaults;                 % FieldTrip + external/stats
addpath(fullfile(ftdir,'external','eeglab'));
addpath(sasdir);
addpath(fullfile(sasdir,'CARACAS'));
addpath(fullfile(sasdir,'CARACAS','heart_functions'));
addpath(fullfile(sasdir,'eeglab')); addpath(genpath(fullfile(sasdir,'eeglab','functions')));
addpath(fullfile(sasdir,'eeglab','plugins','firfilt'));
addpath(shimdir);                            % zscore (+ prctile, rep2struct)
ft_warning('off','FieldTrip:dataContainsNaN');
```

Don't `addpath(genpath(sasdir))` — it would put `private/` and both CARACAS versions' helpers in odd orders. Don't add a full EEGLAB install alongside SASICA's minimal copy (duplicate `eeg_*` functions; `ft_SASICA` explicitly removes other EEGLABs via `rm_frompath eeglab`).

## Building the EEG struct from FieldTrip `comp` (what works around the `ft_SASICA` bug)

Required: `nbchan`, `trials`, `pnts`, `srate`, `xmin`, `xmax`, `times`, `icaact` (ncomp × pnts), `icawinv` (= `comp.topo`), `icaweights` (= `comp.unmixing`), `icasphere` (= eye), `chanlocs` with X/Y/Z (then `convertlocs(...,'cart2all')`), `data` (any consistent nbchan × pnts; `icawinv*icaact` is fine), `icachansind`. For EGI HydroCel 257 the template is `fieldtrip/template/electrode/GSN-HydroCel-257.sfp`; `VREF` has no entry (the wrapper maps it to `Cz`).

## Running headless

- From Python: `subprocess.run([MATLAB_BIN, "-batch", "addpath(...); fn(args)"])` (pattern in `hr/caracas/run.py`).
- Via the MATLAB MCP server: fine for quick `which`/small evaluations, but avoid `restoredefaultpath` inside the MCP session — a call doing it hung for > 2 min on 2026-09-30 (probable cause: it removes the MCP toolbox's own path; not confirmed). Use `matlab -batch` for anything that resets the path.
- Fresh MATLAB: first `ft_preprocessing` call costs 10–50 s warm-up.

## Measured behaviour

- `heart_peak_detect` with SASICA `cfg_peak` on synthetic 70 bpm ECG (2 min, 250 Hz): 139/139 beats, sk 3.63, ku 18.85, bpm 69.5.
- Pure noise, 10 min at 250 Hz: 0.6 s per call.
- NaN spread from 1100-sample padding at 250 Hz: +224 samples (hp firws order 414, lp order 34).
- fs = 180 Hz with defaults → `Lowpass filter frequency too high. Set cfg.lpfreq below 90`.
