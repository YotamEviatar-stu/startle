# PsPM respiration GLM — ground truth

Source: `defaults.glm` table in `src/pspm_init.m` and the three basis
function files `pspm_bf_rprf_e.m`, `pspm_bf_rarf_e.m`, `pspm_bf_rfrrf_e.m`
in bachlab/PsPM. This is what `pspm_glm.m` actually loads when
`model.modality` is `'rp'`, `'ra'`, or `'rfr'` — i.e. the literal defaults
a user gets if they don't override anything.

## Modality-specific defaults table

| modality | modelspec | basis function | derivative arg | filter: lp / lp-order / hp / hp-order | down | direction |
|---|---|---|---|---|---|---|
| `ra` | `ra_e` (evoked) | `pspm_bf_rarf_e` | 1 (yes) | 1 Hz / 1 / 0.001 Hz / 1 | 10 Hz | uni |
| `ra` | `ra_fc` (fear-conditioning) | `pspm_bf_rarf_fc` | 1 (yes) | 2 Hz / 6 / 0.01 Hz / 6 | 10 Hz | bi |
| `rp` | `rp_e` (evoked) | `pspm_bf_rprf_e` | 0 (no) | 1 Hz / 1 / 0.01 Hz / 1 | 10 Hz | uni |
| `rfr` | `rfr_e` (evoked) | `pspm_bf_rfrrf_e` | 1 (yes) | 1 Hz / 1 / 0.001 Hz / 1 | 10 Hz | uni |

Read verbatim from `pspm_init.m`:

```matlab
% GLM for RA (evoked)
defaults.glm(end+1) = struct(...
  'modality',  'ra', 'modelspec', 'ra_e', ...
  'cbf', struct('fhandle', @pspm_bf_rarf_e, 'args', 1), ...
  'filter', struct('lpfreq', 1, 'lporder', 1, 'hpfreq', 0.001, 'hporder', 1, 'down', 10, 'direction', 'uni'), ...
  'default', 0);
% GLM for RP (evoked)
defaults.glm(end+1) = struct(...
  'modality',  'rp', 'modelspec', 'rp_e', ...
  'cbf', struct('fhandle', @pspm_bf_rprf_e, 'args', 0), ...
  'filter', struct('lpfreq', 1, 'lporder', 1, 'hpfreq', 0.01, 'hporder', 1, 'down', 10, 'direction', 'uni'), ...
  'default', 0);
% GLM for RFR (evoked)
defaults.glm(end+1) = struct(...
  'modality',  'rfr', 'modelspec', 'rfr_e', ...
  'cbf', struct('fhandle', @pspm_bf_rfrrf_e, 'args', 1), ...
  'filter', struct('lpfreq', 1, 'lporder', 1, 'hpfreq', 0.001, 'hporder', 1, 'down', 10, 'direction', 'uni'), ...
  'default', 0);
```

**The critical, easy-to-miss point: RP gets a different high-pass cutoff
(0.01 Hz) than RA and RFR (0.001 Hz).** PsPM does not use one shared
"sensitivity filter" across all three channels — RP is deliberately
filtered less aggressively at the low-frequency end than RA/RFR. A
pipeline that applies a single shared high-pass (e.g. one `GLM_FINAL_HP`
constant for RP, RA, and RFR alike) is collapsing a real, documented
per-modality distinction PsPM makes on purpose. This is worth flagging
even though it's a 10x difference in a very low cutoff (0.01 vs 0.001 Hz)
that might look negligible — it changes how much slow drift survives in
the RP series specifically, which directly affects RP-based amplitude
estimates.

Also note: `ra_fc` (fear-conditioning modelspec) uses a **bidirectional**,
6th-order filter — a completely different filter design from `ra_e`. If
comparing an "evoked response" pipeline against PsPM, compare against the
`_e` variants, not `_fc` — mixing them (e.g. using `ra_fc`'s filter
settings with `ra_e`'s basis function) is not a documented PsPM
configuration.

## Basis functions (canonical response shapes)

All three are Gaussian bumps over a fixed window from -10 s to +30 s
relative to cycle/event onset, evaluated at the model's time resolution
`td`, then optionally paired with a time-derivative regressor,
orthogonalized (`spm_orth`) and peak-normalized to unit range:

```matlab
x = (start:td:stop-td)';      % start = -10, stop = 30
bs = exp(-(x-mu).^2 ./ (2*sigma^2));
if bf_type == 1
  bs = [bs [diff(bs); 0]];    % append derivative regressor
end
bs = spm_orth(bs);                              % orthogonalize regressors
bs = bs ./ repmat((max(bs)-min(bs)), size(bs,1), 1);  % normalize to unit range
```

| response function | file | mu (peak latency, s) | sigma (s) | default derivative? |
|---|---|---|---|---|
| RP-RF (respiration period) | `pspm_bf_rprf_e.m` | 4.2 | 1.65 | no (`bf_type` default 0) |
| RA-RF (respiration amplitude) | `pspm_bf_rarf_e.m` | 8.07 | 3.74 | yes (`bf_type` default 1) |
| RFR-RF (respiratory flow rate) | `pspm_bf_rfrrf_e.m` | 6.0 | 3.23 | yes (`bf_type` default 1) |

Reference: Bach DR, Gerster S, Tzovara A, Castegnetti G (2016). A linear
model for event-related respiration responses. J Neurosci Methods, 270,
147-155. (RA fear-conditioning variant: Castegnetti G, Tzovara A, Staib M,
Gerster S, Bach DR (2017). Assessing fear learning via conditioned
respiratory amplitude responses. Psychophysiology, 54, 215-223.)

**Interpretation ordering matters**: RP peaks earliest (~4.2 s) and is
expected to *decrease* (deceleration of breathing / longer periods read as
positive RP but the physiological direction is documented as slowing),
RFR peaks in the middle (~6 s), RA peaks latest (~8.07 s) as the deepest
breath. If a pipeline's own `(tau, sigma)` values don't match this table,
or applies the RA basis function's timing to what it's calling "RP", that
is a direct, checkable mismatch — not a judgment call.

## GLM mechanics (`pspm_glm.m`) worth checking against

- **Basis set convolution + orthogonalized derivative** — if a pipeline
  uses a Gaussian bump alone with no derivative regressor for RA or RFR,
  it has dropped a component PsPM includes by default (`args=1`).
- **`model.latency`**: PsPM defaults to `'fixed'` latency (the basis
  function is convolved at the literal event onset). `'free'` latency
  requires a `model.window` and is a materially different, opt-in model
  (dictionary matching search for best-fit latency) — don't treat a
  fixed-latency pipeline as "wrong" for not doing this; it's the default,
  not a requirement.
- **`model.centering`**: mean-centering of the convolved design matrix
  defaults to **on** (`centering=1`). Only SPS models turn this off.
- **`model.missing`**: epochs to exclude are specified in *seconds*, as
  file-level epochs, and passed to the GLM rather than pre-filtered out of
  the input signal — i.e. PsPM's own rejection model integrates missing
  data into the design matrix rather than deleting samples beforehand.
