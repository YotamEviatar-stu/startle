# PsPM respiration preprocessing — ground truth

Source: `src/pspm_resp_pp.m` in bachlab/PsPM (develop branch). This is the
function that turns a raw respiration trace (bellows or cushion transducer)
into respiration period (RP), respiration amplitude (RA), respiratory flow
rate (RFR), and respiration cycle timestamps (RS). Reference: Bach DR,
Gerster S, Tzovara A, Castegnetti G (2016). A linear model for event-related
respiration responses. J Neurosci Methods, 270, 174-155.

Everything below is transcribed from the actual function, not a summary —
quote line numbers/values directly when comparing against Airflow code.

## Stage 1 — filter mean-centred raw data

```matlab
filt.sr        = data.header.sr;   % native sample rate
filt.lpfreq    = 0.6;
filt.lporder   = 1;
filt.hpfreq    = .01;
filt.hporder   = 1;
filt.direction = 'bi';             % bidirectional (filtfilt, zero-phase)
filt.down      = 10;                % downsample target, Hz
[sts, newresp, newsr] = pspm_prepdata(resp - mean(resp,"omitnan"), filt);
```

Key details:
- Mean-centering happens **before** filtering, on the raw trace, once.
- Low-pass and high-pass are two **separate** first-order Butterworth
  filters, each run through `pspm_filtfilt` (zero-phase, forward-backward).
  This is *not* a single combined 2nd-order bandpass design — it's a
  low-pass filtfilt pass followed by a high-pass filtfilt pass. See
  `pspm_prepdata.m`: "the order for bandpass and bandstop filters is equal
  to order = lporder + hporder" is a **developer note about nominal filter
  order bookkeeping**, not a statement that PsPM designs a joint bandpass.
  If an implementation designs one `scipy.signal.butter(N=2, [0.01,0.6],
  btype='band')` instead of two cascaded 1st-order passes, the frequency
  response differs (steeper transition band, different phase behavior)
  even though "order 2" sounds equivalent on paper.
- Downsampling to `filt.down = 10` Hz happens inside `pspm_prepdata` after
  filtering (anti-aliased by the 0.6 Hz low-pass already applied).

## Stage 2 — median filter

```matlab
newresp = medfilt1(newresp, ceil(newsr) + 1);
```

Window length is `ceil(newsr) + 1` **samples**, where `newsr` is the
sample rate *after* downsampling (≈10 Hz), not the native rate. So the
window is ≈11 samples ≈ 1.1 s at 10 Hz — deliberately expressed in
samples, not a fixed second value, so it scales if `filt.down` changes.

## Stage 3 — cycle (breath) detection

Two transducer types, two different zero-crossing rules:

```matlab
if strcmpi(options.systemtype, 'bellows')
  % find pos/neg zero crossings
  respstamp = find(diff(sign(newresp)) == -2)/newsr;
elseif strcmpi(options.systemtype, 'cushion')
  % find neg/pos zero crossings of the FIRST DERIVATIVE
  diffresp = diff(newresp);
  foo = diff(sign(diffresp));
  zero1 = find(foo == 2);
  indx = find(foo ~= 0);
  neighbour_sums = conv(foo(indx), [1 1], 'valid');
  pairs = find(neighbour_sums == 2);
  zero2 = ceil(mean([indx(pairs + 1), indx(pairs)], 2));
  respstamp = sort([zero1;zero2] + 1)/newsr;
end
```

Bellows = increased air flow on inspiration → detect on the signal itself
(falling zero-crossing). Cushion = increased pressure on inspiration →
detect on the signal's derivative (a peak in the raw trace, i.e. a
zero-crossing of the slope). **Airflow sensors are flow transducers, i.e.
bellows-type physics** — the direct zero-crossing rule is the correct one
to mirror, not the derivative rule.

### Why the raw-vs-filtered separation (Stage 1 vs Stage 5) matters *more* for an
### airflow signal than for a classic bellows/chest-strap signal

A flow signal is, physically, close to the **derivative** of a volume/chest-
circumference signal (flow ≈ d(volume)/dt). Differentiating a smooth
waveform pushes energy toward *higher* frequencies — so a flow trace
carries genuinely more high-frequency content than the slow, rounded
bellows signal PsPM's defaults (0.6 Hz lowpass for detection) were tuned
against. Peak inspiratory flow in particular tends to be a **sharp, early-
in-breath feature**, not a slow rounded peak like chest circumference.

Consequence: the 0.6 Hz detection filter (`filt.lpfreq = 0.6` in Stage 1)
is more likely to be clipping *real physiology* — not just noise — when
applied to a flow-type signal, compared to a bellows trace. This raises
the stakes on Stage 5's rule (measure RA/RFR on the untouched `resp`
variable, never on `newresp`): for an airflow sensor specifically, skipping
that separation (i.e. measuring amplitude on the filtered/detection copy,
the way `airflow_glm.py`'s `detect_cycles()` currently does on `raw_z`)
risks losing a larger fraction of the true peak height than it would for
the bellows signals PsPM was originally validated on. This is a reason to
treat the RA/RFR-on-filtered-signal finding (see the audit workflow in
`SKILL.md`) as higher priority for airflow data than it might be for other
transducer types — not a reason to change which systemtype/detection rule
to use (that part — direct zero-crossing, no derivative — is still
correct for a flow signal).

## Stage 4 — implausible-cycle rejection

```matlab
ibi = diff(respstamp);
indx = find(ibi < 1);
respstamp(indx + 1) = [];
```

Only a **hard floor of 1 second** between cycle onsets is enforced here
(cycles faster than 60 bpm are discarded as noise). There is no upper IBI
bound enforced at this stage — the diagnostic plot flags `ibi > 10 s`
(comment: "here we flag values outside 1-10 s breathing period", citing
Schmidt/Thews normal range of 10-18 breaths/min = 3.3-6 s/cycle) but this
is a **plotting flag only**, not an exclusion. If a pipeline rejects
trials on an upper-IBI/lower-rate bound, that is an addition beyond
`pspm_resp_pp.m`, not a PsPM-documented rule — defensible, but should be
labeled as a project-specific artifact gate, not "the PsPM rate ceiling."

## Stage 5 — compute RP / RA / RFR per cycle

```matlab
% RP: respiration period
respdata = diff(respstamp);                       % seconds between cycle onsets

% RA: respiration amplitude
win = ceil(respstamp(k)*sr) : ceil(respstamp(k+1)*sr);   % on native-rate resp
respdata(k) = range(resp(win));                    % max-min of RAW resp in window

% RFR: respiratory flow rate
respdata(k) = range(resp(win)) / ibi(k);            % RA / RP for that cycle
```

Two details easy to get wrong:
- RA/RFR are computed on the **native sample rate raw signal** (`resp`,
  before mean-centering/filtering/downsampling), using `range()` = max−min
  over each breath's sample window — not on the filtered/downsampled trace.
- The window bounds are the **current cycle onset to the next cycle
  onset**, i.e. RA is a property of the completed breath, assigned to it
  after the fact.

## Stage 6 — assignment and interpolation

```matlab
% assign rp/ra/RFR to the *following* zero crossing, then linearly
% interpolate onto a fresh regular time grid at the requested sr
writedata = interp1(respstamp(2:end), respdata, newt, 'linear');
```

Each RP/RA/RFR value is anchored to the **end** of the cycle it describes
(the second of the two zero-crossings that bound it), not the start. Off-
grid extrapolation at the edges uses nearest-neighbour, not linear
extrapolation, to avoid runaway values.

## What this stage does *not* do

- No artifact/amplitude-based cycle rejection (e.g. no "reject cycles with
  RA more than N SD from session median") — PsPM leaves this to the
  downstream GLM's own missing-epoch mechanism (`model.missing`), not to
  `pspm_resp_pp`.
- No baseline correction / z-scoring here — that happens (optionally) in
  the GLM stage per modality (see `pspm-glm-respiration.md`).
