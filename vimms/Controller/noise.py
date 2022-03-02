import itertools
from abc import abstractmethod

from vimms.Controller.roi import RoiController


class NoiseEstimator():
    @abstractmethod
    def estimate_noise(self, roi): pass


class IdentityEstimator():
    def estimate_noise(self, roi): return 1.0


# It's a peak if it's intensity is > X for N scans in a row.
class ThresholdEstimator(NoiseEstimator):
    def __init__(self, intensity_threshold, count_threshold):
        self.intensity_threshold = intensity_threshold
        self.count_threshold = count_threshold

    def estimate_noise(self, roi):
        aboves = [intensity > self.intensity_threshold for intensity in
                  roi.intensity_list]
        return float(
            max(itertools.accumulate(aboves, lambda a, b: a + 1 if b else 0,
                                     initial=0)) >= self.count_threshold)


# It's a peak if we observe intensity increasing (maybe a rolling average)
# for N scans in a row
class IncreaseEstimator(NoiseEstimator):
    def __init__(self, count_threshold):
        self.count_threshold = count_threshold

    def estimate_noise(self, roi):
        increases = (roi.intensity_list[i + 1] - roi.intensity_list[i] > 0
                     for i in range(len(roi.intensity_list) - 1))
        return float(
            max(itertools.accumulate(increases, lambda a, b: a + 1 if b else 0,
                                     initial=0)) >= self.count_threshold)


# class OracleEstimator(NoiseEstimator):

class NoiseController(RoiController):
    def __init__(self, ionisation_mode, isolation_width, mz_tol,
                 min_ms1_intensity, min_roi_intensity,
                 min_roi_length, N, noise_estimator, rt_tol=10,
                 min_roi_length_for_fragmentation=1,
                 ms1_shift=0,
                 advanced_params=None):
        super().__init__(
            ionisation_mode, isolation_width, mz_tol, min_ms1_intensity,
            min_roi_intensity,
            min_roi_length, N, rt_tol=rt_tol,
            min_roi_length_for_fragmentation=min_roi_length_for_fragmentation,
            ms1_shift=ms1_shift, advanced_params=advanced_params
        )

        # given a RoI, returns probability that it is noise
        self.noise_estimator = noise_estimator

    def _get_scores(self):
        p = [self.noise_estimator.estimate_noise(r) for r in self.live_roi]
        return self._get_top_N_scores(self._get_dda_scores() * p)
