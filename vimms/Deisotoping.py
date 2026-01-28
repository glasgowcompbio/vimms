from dataclasses import dataclass
from collections import deque
from typing import Iterable, List, Tuple

import numpy as np

from vimms.Common import C13_MZ_DIFF, NATURAL_ISOTOPES


@dataclass(frozen=True)
class IsotopeCluster:
    monoisotopic_mz: float
    charge: int
    peak_indices: Tuple[int, ...]


class Deisotoper:
    def __init__(
        self,
        ppm_tolerance=10.0,
        max_charge=3,
        min_isotopes=2,
        isotope_mass_diffs=None,
        max_relative_intensity_increase=1.5,
        max_relative_intensity_increase_heavy=3.0,
        heavy_isotope_threshold=1.5,
    ):
        self.ppm_tolerance = ppm_tolerance
        self.max_charge = max_charge
        self.min_isotopes = min_isotopes
        self.isotope_mass_diffs = (
            tuple(isotope_mass_diffs)
            if isotope_mass_diffs is not None
            else self._default_isotope_mass_diffs()
        )
        self.max_relative_intensity_increase = max_relative_intensity_increase
        self.max_relative_intensity_increase_heavy = max_relative_intensity_increase_heavy
        self.heavy_isotope_threshold = heavy_isotope_threshold

    def deisotope(self, peaks: Iterable[Tuple[float, float]]) -> List[IsotopeCluster]:
        peaks = np.array(list(peaks), dtype=float)
        if peaks.size == 0:
            return []

        mzs = peaks[:, 0]
        intensities = peaks[:, 1]
        order = np.argsort(mzs)
        mzs = mzs[order]
        intensities = intensities[order]

        assigned = np.full(len(mzs), False)
        clusters = []

        for idx in range(len(mzs)):
            if assigned[idx]:
                continue
            mz = mzs[idx]
            charge = self._guess_charge(mzs, mz, idx)
            cluster_indices = self._grow_cluster(mzs, intensities, idx, charge)

            if len(cluster_indices) >= self.min_isotopes:
                for ci in cluster_indices:
                    assigned[ci] = True
                clusters.append(
                    IsotopeCluster(
                        monoisotopic_mz=mz,
                        charge=charge,
                        peak_indices=tuple(order[cluster_indices]),
                    )
                )

        return clusters

    def _guess_charge(self, mzs: np.ndarray, mz: float, idx: int) -> int:
        # Prefer charge assignments that match the 13C spacing.
        best_charge = None
        best_error = None
        for charge in range(1, self.max_charge + 1):
            target = mz + C13_MZ_DIFF / charge
            match_idx = self._find_peak(mzs, target, idx + 1)
            if match_idx is None:
                continue
            error = self._ppm_error(mzs[match_idx], target)
            if best_error is None or error < best_error:
                best_error = error
                best_charge = charge
        if best_charge is not None:
            return best_charge

        best_charge = 1
        best_match = None
        for charge in range(1, self.max_charge + 1):
            match = self._find_isotope_peak(mzs, idx + 1, mz, charge, isotope_idx=1)
            if match is None:
                continue
            _, _, delta = match
            if best_match is None or delta < best_match:
                best_match = delta
                best_charge = charge
        return best_charge

    def _grow_cluster(
        self, mzs: np.ndarray, intensities: np.ndarray, start_idx: int, charge: int
    ) -> List[int]:
        # Build a connected component of peaks linked by any single-isotope mass
        # difference. This avoids splitting fine-structure isotope patterns into
        # multiple clusters.
        cluster = {start_idx}
        queue = deque([start_idx])

        while queue:
            current_idx = queue.popleft()
            current_mz = mzs[current_idx]
            current_intensity = intensities[current_idx]

            for diff in self.isotope_mass_diffs:
                target = current_mz + diff / charge
                match_idx = self._find_peak(mzs, target, current_idx + 1)
                if match_idx is None or match_idx in cluster:
                    continue

                max_increase = self.max_relative_intensity_increase
                if diff >= self.heavy_isotope_threshold:
                    max_increase = self.max_relative_intensity_increase_heavy
                if intensities[match_idx] > current_intensity * max_increase:
                    continue

                cluster.add(match_idx)
                queue.append(match_idx)

        return sorted(cluster)

    def _find_isotope_peak(
        self, mzs: np.ndarray, start_idx: int, base_mz: float, charge: int, isotope_idx: int
    ) -> Tuple[int, float, float] | None:
        best = None
        for diff in self.isotope_mass_diffs:
            target = base_mz + (diff / charge) * isotope_idx
            match_idx = self._find_peak(mzs, target, start_idx)
            if match_idx is None:
                continue
            delta = abs(mzs[match_idx] - target)
            if best is None or delta < best[2]:
                best = (match_idx, diff, delta)
        return best

    def _find_peak(self, mzs: np.ndarray, target: float, start_idx: int) -> int | None:
        if start_idx >= len(mzs):
            return None
        left = start_idx
        right = len(mzs) - 1
        while left <= right:
            mid = (left + right) // 2
            if mzs[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        candidates = []
        for idx in (left - 2, left - 1, left, left + 1, left + 2):
            if start_idx <= idx < len(mzs):
                candidates.append(idx)

        best_idx = None
        best_error = None
        for idx in candidates:
            error = self._ppm_error(mzs[idx], target)
            if error > self.ppm_tolerance:
                continue
            if best_error is None or error < best_error:
                best_error = error
                best_idx = idx

        return best_idx

    @staticmethod
    def _ppm_error(mz: float, target: float) -> float:
        return abs(mz - target) / target * 1e6

    @staticmethod
    def _default_isotope_mass_diffs(
        min_abundance: float = 0.0001, max_shift: float = 4.0
    ) -> Tuple[float, ...]:
        diffs = {round(C13_MZ_DIFF, 6)}
        for isotopes in NATURAL_ISOTOPES.values():
            if len(isotopes) <= 1:
                continue
            mono_mass = isotopes[0][0]
            for mass, abundance in isotopes[1:]:
                if abundance < min_abundance:
                    continue
                diff = mass - mono_mass
                if 0 < diff <= max_shift:
                    diffs.add(round(diff, 6))
        return tuple(sorted(diffs))
