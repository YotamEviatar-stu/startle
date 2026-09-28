import numpy as np

from .fft_smooth import fft_smooth
from .extrema import find_respiratory_extrema
from .onsets_pauses import find_respiratory_pauses_and_onsets
from .offsets import find_respiratory_offsets
from .durations import find_breath_durations
from .volumes import find_respiratory_volumes
from .features import get_secondary_features
from .erp import create_respiratory_erp_matrix

SUPPORTED_DATA_TYPES = ("humanAirflow", "humanBB", "rodentAirflow", "rodentThermocouple")
AIRFLOW_TYPES = ("humanAirflow", "rodentAirflow")


class BreathMetrics:

    def __init__(self, resp, srate, data_type="humanAirflow", smooth_win_ms=None):
        resp = np.asarray(resp, dtype=float).ravel()
        srate = float(srate)

        if data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError("data_type must be one of %s" % (SUPPORTED_DATA_TYPES,))
        if not (20.0 <= srate <= 5000.0):
            raise ValueError("sampling rates outside 20-5000 Hz are not supported")

        self.raw_respiration = resp
        self.srate = srate
        self.data_type = data_type
        self.time = np.arange(1, resp.size + 1) / srate

        if smooth_win_ms is None:
            smooth_win_ms = 50.0 if data_type.startswith("human") else 10.0
        win = int(np.floor((srate / 1000.0) * smooth_win_ms))
        self.smoothed_respiration = fft_smooth(resp, win)

        self.baseline_corrected_respiration = None
        self.inhale_peaks = None
        self.exhale_troughs = None
        self.peak_inspiratory_flows = None
        self.trough_expiratory_flows = None
        self.inhale_onsets = None
        self.exhale_onsets = None
        self.inhale_offsets = None
        self.exhale_offsets = None
        self.inhale_pause_onsets = None
        self.exhale_pause_onsets = None
        self.inhale_time_to_peak = None
        self.exhale_time_to_trough = None
        self.inhale_durations = None
        self.exhale_durations = None
        self.inhale_pause_durations = None
        self.exhale_pause_durations = None
        self.inhale_volumes = None
        self.exhale_volumes = None
        self.secondary_features = None
        self.statuses = None
        self.erp_matrix = None
        self.erp_x_axis = None
        self.erp_trial_events = None
        self.erp_rejected_events = None

    def _resp(self):
        if self.baseline_corrected_respiration is None:
            return self.smoothed_respiration
        return self.baseline_corrected_respiration

    def correct_to_baseline(self, method="sliding", z_score=False, window_sec=60.0):
        if method == "none":
            out = self.smoothed_respiration.copy()
        else:
            x = self.smoothed_respiration
            t = np.arange(x.size, dtype=float)
            coef = np.polyfit(t, x, 1)
            detrended = x - np.polyval(coef, t)
            demeaned = detrended - detrended.mean()

            if method == "simple":
                out = demeaned
            elif method == "sliding":
                out = detrended - fft_smooth(detrended, int(np.floor(self.srate * window_sec)))
            else:
                raise ValueError("method must be 'none', 'simple' or 'sliding'")

        if z_score:
            out = (out - out.mean()) / out.std(ddof=1)

        self.baseline_corrected_respiration = out
        return self

    def find_extrema(self, simplify=True, sw_sizes_ms=None, decision_threshold=0):
        resp = self._resp()
        if sw_sizes_ms is None:
            sw = None
        else:
            sw = [int(np.floor(s * self.srate / 1000.0)) for s in sw_sizes_ms]

        peaks, troughs = find_respiratory_extrema(resp, self.srate,
                                                  decision_threshold, sw)
        if simplify:
            n = min(peaks.size, troughs.size)
            peaks, troughs = peaks[:n], troughs[:n]

        if self.data_type in AIRFLOW_TYPES:
            self.inhale_peaks = peaks
            self.exhale_troughs = troughs
            self.peak_inspiratory_flows = resp[peaks]
            self.trough_expiratory_flows = resp[troughs]
        elif self.data_type == "humanBB":
            if troughs.size and peaks.size and troughs[0] > peaks[0]:
                peaks, troughs = peaks[1:], troughs[:-1]
            self.inhale_onsets = troughs
            self.exhale_onsets = peaks
        else:
            self.inhale_onsets = peaks
            self.exhale_onsets = troughs
        return self

    def find_onsets_and_pauses(self, n_bins=None):
        resp = self._resp()
        if n_bins is None:
            n_bins = max(int(np.floor(self.srate / 100.0)), 20)

        (self.inhale_onsets, self.exhale_onsets,
         self.inhale_pause_onsets, self.exhale_pause_onsets) = \
            find_respiratory_pauses_and_onsets(resp, self.inhale_peaks,
                                               self.exhale_troughs, n_bins)

        self.inhale_time_to_peak = (self.inhale_peaks - self.inhale_onsets) / self.srate
        n = self.exhale_troughs.size
        self.exhale_time_to_trough = (self.exhale_troughs[:n] - self.exhale_onsets[:n]) / self.srate
        return self

    def find_offsets(self):
        self.inhale_offsets, self.exhale_offsets = find_respiratory_offsets(
            self._resp(), self.inhale_onsets, self.exhale_onsets,
            self.inhale_pause_onsets, self.exhale_pause_onsets)
        return self

    def find_durations(self):
        (self.inhale_durations, self.exhale_durations,
         self.inhale_pause_durations, self.exhale_pause_durations) = \
            find_breath_durations(self.srate, self.inhale_onsets, self.inhale_offsets,
                                  self.exhale_onsets, self.exhale_offsets,
                                  self.inhale_pause_onsets, self.exhale_pause_onsets)
        return self

    def find_volumes(self):
        self.inhale_volumes, self.exhale_volumes = find_respiratory_volumes(
            self._resp(), self.srate, self.inhale_onsets, self.exhale_onsets,
            self.inhale_offsets, self.exhale_offsets)
        return self

    def get_secondary_features(self):
        self.secondary_features = get_secondary_features(self)
        return self

    def estimate_all_features(self, baseline_method="sliding", z_score=False,
                              simplify=True, n_bins=None):
        self.correct_to_baseline(baseline_method, z_score)
        self.find_extrema(simplify)
        if self.data_type in AIRFLOW_TYPES:
            self.find_onsets_and_pauses(n_bins)
            self.find_offsets()
            self.find_durations()
            self.find_volumes()
        self.get_secondary_features()
        return self

    def calculate_erp(self, event_array, pre_ms, post_ms, append_nans=False):
        pre = int(round(pre_ms * self.srate / 1000.0))
        post = int(round(post_ms * self.srate / 1000.0))
        (self.erp_matrix, self.erp_trial_events, self.erp_rejected_events,
         _, _) = create_respiratory_erp_matrix(self._resp(), event_array,
                                               pre, post, append_nans)
        self.erp_x_axis = np.arange(-pre, post + 1) / self.srate
        return self
