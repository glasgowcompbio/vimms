"""Simple linear retention time model with optional per-chemical noise."""

from __future__ import annotations

import copy
import numpy as np
from typing import Sequence, Optional


class LinearColumn:
    """Map hydrophobicity to retention time via a linear relationship."""

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
        self.min_rt = float(min_rt)
        self.max_rt = float(max_rt)
        self.noise_sd = float(noise_sd)
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        self.slope = (self.max_rt - self.min_rt) / (self.max_logp - self.min_logp)
        self.intercept = self.min_rt - self.slope * self.min_logp

    # ------------------------------------------------------------------
    def hydrophobicity_to_rt(self, logp: float, *, noisy: bool = True) -> float:
        base_rt = self.intercept + self.slope * logp
        if noisy and self.noise_sd > 0:
            base_rt += float(self._rng.normal(0.0, self.noise_sd))
        return float(base_rt)

    def apply(self, chemicals: Sequence) -> list:
        """Return deep copies of ``chemicals`` with RT drift applied."""
        new = []
        for chem in chemicals:
            dup = copy.deepcopy(chem)
            dup.rt = self.hydrophobicity_to_rt(dup.rt)
            new.append(dup)
        return new

    # Backwards compat -------------------------------------------------
    def get_dataset(self, chemicals: Sequence) -> list:
        return self.apply(chemicals)


__all__ = ["LinearColumn"]
