"""
vimms.rt.column_drift
~~~~~~~~~~~~~~~~~~~~~

Run-to-run retention-time drift models that wrap :class:`vimms.rt.linear_column.LinearColumn`.

The base class encapsulates the bookkeeping needed to turn an
(additive) intercept shift and a (multiplicative) slope scale into the
two ``min_rt`` / ``max_rt`` numbers that :class:`LinearColumn` expects.
Sub-classes decide how those two drift parameters are obtained.

Copyright (c) 2025
Released under the MIT licence, see the project's LICENCE file.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Sequence, Optional, Tuple

from vimms.rt.linear_column import LinearColumn


class BaseColumnDriftModel(ABC):
    """Abstract base class for objects that supply *one* appropriately
    drifted :class:`LinearColumn` per simulated LC-MS injection.

    Parameters
    ----------
    min_logp, max_logp
        Domain of the hydrophobicity scale handed to ``LinearColumn``.
    min_rt, max_rt
        Nominal (drift-free) gradient start/stop times in minutes.
    noise_sd
        Per-chemical Gaussian noise passed straight through to
        ``LinearColumn``.
    random_state
        Anything accepted by :pyfunc:`numpy.random.default_rng`.
    """

    def __init__(
        self,
        min_logp: float,
        max_logp: float,
        min_rt: float,
        max_rt: float,
        noise_sd: float = 0.0,
        random_state: Optional[int | np.random.RandomState | np.random.Generator] = None,
    ) -> None:
        self.min_logp = float(min_logp)
        self.max_logp = float(max_logp)
        self.noise_sd = float(noise_sd)

        self._base_slope: float = (max_rt - min_rt) / (max_logp - min_logp)
        self._base_intercept: float = min_rt - self._base_slope * min_logp

        # Using NumPy’s new Generator API keeps ViMMS reproducible
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    @abstractmethod
    def _sample_drift(self) -> Tuple[float, float]:
        """Return a tuple ``(intercept_shift, slope_scale)`` for a single run.

        * **intercept_shift** is *additive* and expressed in **minutes**.
        * **slope_scale** is *multiplicative* (1.0 means “no change”).
        """
        raise NotImplementedError

    def make_column(self) -> LinearColumn:
        """Create and return a :class:`LinearColumn` whose slope and intercept
        include one draw from this model’s drift distribution."""
        intercept_shift, slope_scale = self._sample_drift()

        slope = self._base_slope * slope_scale
        intercept = self._base_intercept + intercept_shift

        min_rt = intercept + slope * self.min_logp
        max_rt = intercept + slope * self.max_logp

        column_seed = self._rng.integers(2**32)

        return LinearColumn(
            min_logp=self.min_logp,
            max_logp=self.max_logp,
            min_rt=min_rt,
            max_rt=max_rt,
            noise_sd=self.noise_sd,
            random_state=int(column_seed),
        )


class SimulatedDriftModel(BaseColumnDriftModel):
    """Draw run-to-run drift from *Normal* priors."""

    def __init__(
        self,
        min_logp: float,
        max_logp: float,
        min_rt: float,
        max_rt: float,
        *,
        intercept_mu: float = 0.0,
        intercept_sd: float = 0.0,
        slope_mu: float = 1.0,
        slope_sd: float = 0.0,
        noise_sd: float = 0.0,
        random_state: Optional[int | np.random.RandomState | np.random.Generator] = None,
    ) -> None:
        super().__init__(
            min_logp=min_logp,
            max_logp=max_logp,
            min_rt=min_rt,
            max_rt=max_rt,
            noise_sd=noise_sd,
            random_state=random_state,
        )
        self._i_mu = float(intercept_mu)
        self._i_sd = float(intercept_sd)
        self._m_mu = float(slope_mu)
        self._m_sd = float(slope_sd)

    def _sample_drift(self) -> Tuple[float, float]:
        intercept_shift = (
            self._rng.normal(self._i_mu, self._i_sd) if self._i_sd > 0 else self._i_mu
        )
        slope_scale = (
            self._rng.normal(self._m_mu, self._m_sd) if self._m_sd > 0 else self._m_mu
        )
        return float(intercept_shift), float(slope_scale)


class DataDrivenDriftModel(BaseColumnDriftModel):
    """Replay empirically measured drift parameters."""

    def __init__(
        self,
        min_logp: float,
        max_logp: float,
        min_rt: float,
        max_rt: float,
        *,
        intercepts: Sequence[float],
        slopes: Sequence[float],
        noise_sd: float = 0.0,
        random_state: Optional[int | np.random.RandomState | np.random.Generator] = None,
    ) -> None:
        if len(intercepts) != len(slopes):
            raise ValueError("intercepts and slopes must have the same length")

        super().__init__(
            min_logp=min_logp,
            max_logp=max_logp,
            min_rt=min_rt,
            max_rt=max_rt,
            noise_sd=noise_sd,
            random_state=random_state,
        )
        self._intercepts = np.asarray(intercepts, dtype=float)
        self._slopes = np.asarray(slopes, dtype=float)
        self._cursor = 0

    def _sample_drift(self) -> Tuple[float, float]:
        intercept_shift = self._intercepts[self._cursor]
        slope_scale = self._slopes[self._cursor]

        self._cursor += 1
        if self._cursor >= len(self._intercepts):
            self._cursor = 0

        return float(intercept_shift), float(slope_scale)
