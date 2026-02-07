from dataclasses import dataclass
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
        cluster_indices = [start_idx]
        isotope_idx = 1
        prev_intensity = intensities[start_idx]
        while True:
            match = self._find_isotope_peak(
                mzs, cluster_indices[-1] + 1, mzs[start_idx], charge, isotope_idx
            )
            if match is None:
                break
            match_idx, match_diff, _ = match
            max_increase = self.max_relative_intensity_increase
            if match_diff >= self.heavy_isotope_threshold:
                max_increase = self.max_relative_intensity_increase_heavy
            if intensities[match_idx] > prev_intensity * max_increase:
                break
            cluster_indices.append(match_idx)
            prev_intensity = intensities[match_idx]
            isotope_idx += 1
        return cluster_indices

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
        for idx in (left - 1, left, left + 1):
            if 0 <= idx < len(mzs):
                candidates.append(idx)
        for idx in candidates:
            if self._ppm_error(mzs[idx], target) <= self.ppm_tolerance:
                return idx
        return None

    @staticmethod
    def _ppm_error(mz: float, target: float) -> float:
        return abs(mz - target) / target * 1e6

    @staticmethod
    def _default_isotope_mass_diffs(
        min_abundance: float = 0.0005, max_shift: float = 4.0
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


def deisotope_with_ms_deisotope(
    peaks: Iterable[Tuple[float, float]],
    charge_range: Tuple[int, int] = (1, 3),
    averagine: str = "peptide",
    ms1_tolerance: float = 10.0,
):
    """
    Deisotope peaks using the optional ms_deisotope package.

    Args:
        peaks: iterable of (mz, intensity) pairs.
        charge_range: inclusive min/max charge range.
        averagine: averagine model name used by ms_deisotope.
        ms1_tolerance: ppm tolerance for isotopic matching.

    Returns:
        The ms_deisotope deconvolution result object.
    """
    from ms_deisotope.deconvolution import deconvolute_peaks

    peaks_array = np.array(list(peaks), dtype=float)
    if peaks_array.size == 0:
        return []

    return deconvolute_peaks(
        peaks_array,
        charge_range=charge_range,
        averagine=averagine,
        ms1_tolerance=ms1_tolerance,
    )


def deisotope_with_pyopenms(
    peaks: Iterable[Tuple[float, float]],
    fragment_tolerance: float = 10.0,
    fragment_unit_ppm: bool = True,
    min_charge: int = 1,
    max_charge: int = 3,
    keep_only_deisotoped: bool = True,
    min_isopeaks: int = 2,
    max_isopeaks: int = 10,
    make_single_charged: bool = False,
):
    """
    Deisotope peaks using pyopenms Deisotoper on an MS1-like spectrum.

    Args:
        peaks: iterable of (mz, intensity) pairs.
        fragment_tolerance: m/z tolerance for matching isotopic peaks.
        fragment_unit_ppm: whether the tolerance is in ppm.
        min_charge: minimum charge to consider.
        max_charge: maximum charge to consider.
        keep_only_deisotoped: keep only deisotoped peaks in the output.
        min_isopeaks: minimum number of isotopic peaks in a cluster.
        max_isopeaks: maximum number of isotopic peaks in a cluster.
        make_single_charged: convert all features to single charge if True.

    Returns:
        Tuple of (spectrum, peak_mzs, peak_intensities) after deisotoping.
    """
    import pyopenms as oms

    peaks_array = np.array(list(peaks), dtype=float)
    if peaks_array.size == 0:
        return None, np.array([]), np.array([])

    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((peaks_array[:, 0], peaks_array[:, 1]))

    oms.Deisotoper.deisotopeAndSingleCharge(
        spectrum,
        fragment_tolerance,
        fragment_unit_ppm,
        min_charge,
        max_charge,
        keep_only_deisotoped,
        min_isopeaks,
        max_isopeaks,
        make_single_charged,
    )

    mzs, intensities = spectrum.get_peaks()
    return spectrum, np.array(mzs), np.array(intensities)


def deadduct_with_pyopenms(
    feature_map,
    adducts: List[str] | None = None,
    max_charge: int = 3,
    ppm_tolerance: float = 10.0,
):
    """
    De-adduct features using pyopenms MetaboliteAdductDecharger.

    Args:
        feature_map: pyopenms FeatureMap containing detected features.
        adducts: list of adduct strings (e.g. ["[M+H]+", "[M+Na]+"]) or None for defaults.
        max_charge: maximum charge to consider.
        ppm_tolerance: mass tolerance used for matching (ppm).

    Returns:
        De-adducted pyopenms FeatureMap.
    """
    import pyopenms as oms

    decharger = oms.MetaboliteAdductDecharger()
    params = decharger.getParameters()
    params.setValue("mass_error_ppm", ppm_tolerance)
    params.setValue("charge_max", max_charge)
    if adducts:
        adduct_info = oms.AdductInfo()
        adduct_info.setAdducts(adducts)
        decharger.setAdducts(adduct_info)
    decharger.setParameters(params)
    output = oms.FeatureMap()
    decharger.decharge(feature_map, output)
    return output
