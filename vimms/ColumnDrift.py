"""Models run-to-run retention time drift for LC-MS injections."""

import numpy as np
from abc import ABCMeta, abstractmethod

from .Column import LinearColumn


class BaseColumnDriftModel(metaclass=ABCMeta):
    """Base class for objects that provide one drifted ``LinearColumn``."""

    def __init__(self, noise_sd=0.0, random_state=None):
        self.noise_sd = noise_sd
        self._rng = np.random.default_rng(random_state)

    @abstractmethod
    def _sample_drift(self):
        """Return ``(intercept_shift, slope_scale)`` for one injection."""

    def make_column(self, dataset):
        """Return a ``LinearColumn`` with drift for ``dataset``."""
        intercept, scale = self._sample_drift()
        return LinearColumn.from_fixed_offsets(dataset, self.noise_sd, intercept, scale - 1.0)


class SimulatedDriftModel(BaseColumnDriftModel):
    """Draw drift parameters from Normal distributions."""

    def __init__(
        self,
        intercept_mu=0.0,
        intercept_sd=0.0,
        slope_mu=1.0,
        slope_sd=0.0,
        noise_sd=0.0,
        random_state=None,
    ):
        super().__init__(noise_sd, random_state)
        self.intercept_mu = intercept_mu
        self.intercept_sd = intercept_sd
        self.slope_mu = slope_mu
        self.slope_sd = slope_sd

    def _sample_drift(self):
        if self.intercept_sd > 0:
            intercept = self._rng.normal(self.intercept_mu, self.intercept_sd)
        else:
            intercept = self.intercept_mu
        if self.slope_sd > 0:
            slope = self._rng.normal(self.slope_mu, self.slope_sd)
        else:
            slope = self.slope_mu
        return intercept, slope


class DataDrivenDriftModel(BaseColumnDriftModel):
    """Replay empirically observed drift parameters."""

    def __init__(self, intercepts, slopes, noise_sd=0.0, random_state=None):
        if len(intercepts) != len(slopes):
            raise ValueError("intercepts and slopes must have the same length")
        super().__init__(noise_sd, random_state)
        self._intercepts = list(intercepts)
        self._slopes = list(slopes)
        self._cursor = 0

    def _sample_drift(self):
        intercept = self._intercepts[self._cursor]
        slope = self._slopes[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._intercepts)
        return intercept, slope
