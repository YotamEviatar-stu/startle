# Goal log — cardiac IC in ≥75% of sessions

Pass criterion (fixed): `eeg_SASICA` CARACAS block with `SASICA('getdefs')` defaults selects ≥1 IC. ICLabel Heart p logged, not required. IC indices are 0-based.

## c1_readme (partial, 32 of 52 sessions) — 2026-10-01

- **PARTIAL:** raw-data drive unmounted; only sessions processed so far are counted
- **Method:** Extended Infomax, PCA-reduced to round(rank/5) components (CARACAS README example), 1–100 Hz FIR band-pass, average reference, 250 Hz
- **Citation:** CARACAS README example `pop_runica(...,'extended',1,'pca',round(dataRank/5))` (github.com/PierreChampetier/Cardiac_IC_labelling; SASICA CARACAS); extended Infomax: Lee, Girolami & Sejnowski 1999 Neural Comput 11:417, impl. MNE `ICA(method='infomax', extended=True)`; 1 Hz high-pass: Winkler et al. 2015 IEEE EMBC; 1–100 Hz + average reference: ICLabel input spec (Pion-Tonachini et al. 2019 NeuroImage 198:181; mne-icalabel docs); fs 250 Hz > 200 Hz required by heart_peak_detect lpfreq 100 Hz
- **Parameters:** mne 1.8 `read_raw_egi` → pick EEG → `resample(250, 'polyphase')` → `filter(1, 100)` (MNE default firwin) → `set_eeg_reference('average')` → rank = `compute_rank` → `ICA(n_components=round(rank/5), method='infomax', extended=True, max_iter='auto', random_state=97)`; CARACAS = `eeg_SASICA` with `SASICA('getdefs')` CARACAS defaults (sk≥1.0, ku≥5.3, RR≤0.31, Rampl≤0.23, bpm 34–97), all else disabled
- **Pass:** 8/32 = 25.0%
- **Median ICLabel Heart p of passing ICs:** 0.049 (n = 10 ICs)

| session | pass | CARACAS IC(s) | ICLabel Heart p | note |
|---|---|---|---|---|
| AB22/eve | fail |  |  | top-ICLabel IC34 (Heart 0.23) fails RR,bpm: bpm 0.2, sk 11.63, ku 2419.3, RR 1.39, Rampl 0.14 |
| AB22/mor | fail |  |  | top-ICLabel IC23 (Heart 0.66) fails sk,RR,Rampl,bpm: bpm 0.7, sk -0.86, ku 304.8, RR 1.44, Rampl 0.61 |
| AG05/eve | PASS | 36 | 0.00 |  |
| AG05/mor | fail |  |  | top-ICLabel IC50 (Heart 0.14) fails RR,Rampl,bpm: bpm 4.7, sk 2.49, ku 44.0, RR 0.37, Rampl 0.31 |
| AH19/eve | fail |  |  | top-ICLabel IC32 (Heart 0.08) fails RR,Rampl,bpm: bpm 0.6, sk 2.91, ku 276.5, RR 1.31, Rampl 0.32 |
| AH19/mor | fail |  |  | top-ICLabel IC39 (Heart 0.18) fails sk,RR,Rampl,bpm: bpm 7.8, sk 0.86, ku 41.7, RR 0.49, Rampl 0.28 |
| AK12/eve | PASS | 17, 26 | 0.15, 0.05 |  |
| AK12/mor | PASS | 28 | 0.02 |  |
| AS09/eve | fail |  |  | top-ICLabel IC31 (Heart 0.13) fails sk,RR,Rampl,bpm: bpm 0.6, sk -0.66, ku 299.4, RR 0.43, Rampl 0.63 |
| AS09/mor | PASS | 22 | 0.02 |  |
| AS31/eve | fail |  |  | top-ICLabel IC26 (Heart 0.30) fails sk,RR,Rampl,bpm: bpm 0.5, sk 0.93, ku 69.4, RR 1.77, Rampl 0.57 |
| AS31/mor | fail |  |  | top-ICLabel IC38 (Heart 0.14) fails Rampl,bpm: bpm 0.1, sk 4.30, ku 1207.3, RR 0.00, Rampl 1.40 |
| DA01/eve | fail |  |  | top-ICLabel IC45 (Heart 0.37) fails sk,RR,Rampl,bpm: bpm 0.9, sk -0.52, ku 280.0, RR 0.46, Rampl 0.40 |
| DA01/mor | PASS | 22 | 0.22 |  |
| EE10/eve | PASS | 46 | 0.42 |  |
| EE10/mor | PASS | 17 | 0.04 |  |
| ER23/eve | fail |  |  | top-ICLabel IC44 (Heart 0.03) fails Rampl,bpm: bpm 0.1, sk 28.93, ku 3713.3, RR 0.01, Rampl 0.98 |
| ER23/mor | fail |  |  | top-ICLabel IC41 (Heart 0.12) fails RR,bpm: bpm 1.7, sk 6.44, ku 204.2, RR 1.99, Rampl 0.23 |
| ES29/eve | fail |  |  | ES29_eve_20260605_125423: ICA error `FileNotFoundError: [Errno 2] No such file or directory: '/Volumes/My Passport/st`; top-ICLabel IC31 (Heart 0.41) fails sk,RR,Rampl,bpm: bpm 13.1, sk 0.20, ku 5.7, RR 0.66, Rampl 0.57 |
| ES29/mor | fail |  |  | top-ICLabel IC50 (Heart 0.04) fails bpm: bpm 0.0, sk 60.73, ku 10824.3, RR nan, Rampl 0.00 |
| ES32/eve | fail |  |  | top-ICLabel IC30 (Heart 0.09) fails sk: bpm 72.9, sk 0.98, ku 7.4, RR 0.12, Rampl 0.14 |
| ES32/mor | fail |  |  | top-ICLabel IC38 (Heart 0.01) fails sk,Rampl,bpm: bpm 0.6, sk 0.42, ku 484.8, RR 0.18, Rampl 0.93 |
| EV15/eve | fail |  |  | top-ICLabel IC21 (Heart 0.94) fails sk: bpm 55.5, sk 0.60, ku 7.5, RR 0.17, Rampl 0.20 |
| EV15/mor | fail |  |  | top-ICLabel IC27 (Heart 0.19) fails sk,ku,RR: bpm 40.9, sk 0.02, ku 4.3, RR 0.34, Rampl 0.21 |
| LB17/eve | fail |  |  | top-ICLabel IC50 (Heart 0.04) fails sk,Rampl,bpm: bpm 0.1, sk 0.95, ku 1800.1, RR 0.11, Rampl 0.41 |
| LB17/mor | fail |  |  | top-ICLabel IC50 (Heart 0.24) fails sk,RR,Rampl,bpm: bpm 0.9, sk -0.84, ku 461.4, RR 0.45, Rampl 0.80 |
| LG07/eve | PASS | 6, 16 | 0.76, 0.00 |  |
| LG07/mor | fail |  |  | top-ICLabel IC20 (Heart 0.00) fails sk,Rampl,bpm: bpm 0.6, sk -0.18, ku 850.5, RR 0.25, Rampl 0.90 |
| LO21/eve | fail |  |  | top-ICLabel IC40 (Heart 0.20) fails sk,ku,RR,Rampl: bpm 34.9, sk 0.06, ku 5.3, RR 0.45, Rampl 0.38 |
| LO21/mor | fail |  |  | top-ICLabel IC9 (Heart 0.13) fails sk,RR,Rampl,bpm: bpm 7.3, sk 1.00, ku 38.9, RR 0.38, Rampl 0.27 |
| MG14/mor | fail |  |  | MG14_mor_20250928_075517: ICA error `xml.etree.ElementTree.ParseError: junk after document element: line 78, column 6` |
| MH20/mor | fail |  |  | top-ICLabel IC42 (Heart 0.23) fails RR,Rampl,bpm: bpm 1.3, sk 1.16, ku 80.6, RR 1.51, Rampl 0.42 |

Note: ES29/eve's row reflects only its first MFF (`ES29_eve_20260605_125423`, unreadable, known bad file); its real recording `ES29_eve_2_*` was not yet processed when the drive was unmounted, so ES29/eve is pending, not failed. Excluding it: 8/31 = 25.8%.
