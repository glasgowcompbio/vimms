from dataclasses import dataclass
from collections import deque
from typing import Iterable, List, Tuple

import numpy as np
import pyopenms as oms
from ms_deisotope.deconvolution import deconvolute_peaks

from vimms.Common import C13_MZ_DIFF, NATURAL_ISOTOPES, ELECTRON_MASS


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


def deisotope_with_ms_deisotope(
    peaks: Iterable[Tuple[float, float]],
    charge_range: Tuple[int, int] = (1, 3),
    averagine: str = "peptide",
    ms1_tolerance: float = 10.0,
):
    """
    Deisotope peaks using ms_deisotope.

    Args:
        peaks: iterable of (mz, intensity) pairs.
        charge_range: inclusive min/max charge range.
        averagine: averagine model name used by ms_deisotope.
        ms1_tolerance: ppm tolerance for isotopic matching.

    Returns:
        The ms_deisotope deconvolution result object.
    """
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
    keep_only_deisotoped: bool = False,
    min_isopeaks: int = 3,
    max_isopeaks: int = 10,
    make_single_charged: bool = True,
    annotate_charge: bool = False,
    annotate_iso_peak_count: bool = False,
    use_decreasing_model: bool = True,
    start_intensity_check: int = 2,
    add_up_intensity: bool = False,
    annotate_features: bool = False,
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
        annotate_charge: annotate charge in the output if True.
        annotate_iso_peak_count: annotate isotope peak count in the output if True.
        use_decreasing_model: enforce decreasing intensity model if True.
        start_intensity_check: isotope index at which intensity check starts.
        add_up_intensity: add up intensities of isotope peaks if True.
        annotate_features: annotate features in the output if True.

    Returns:
        Tuple of (spectrum, peak_mzs, peak_intensities) after deisotoping.
    """
    peaks_array = np.array(list(peaks), dtype=float)
    if peaks_array.size == 0:
        return None, np.array([]), np.array([])

    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((peaks_array[:, 0], peaks_array[:, 1]))
    spectrum.sortByPosition()

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
        annotate_charge,
        annotate_iso_peak_count,
        use_decreasing_model,
        start_intensity_check,
        add_up_intensity,
        annotate_features,
    )

    mzs, intensities = spectrum.get_peaks()
    return spectrum, np.array(mzs), np.array(intensities)


def deadduct_with_pyopenms(
    feature_map,
    adducts: List[str] | None = None,
    max_charge: int = 3,
    ppm_tolerance: float = 10.0,
    keep_only_backbone: bool = True,
):
    """
    De-adduct features using pyopenms MetaboliteFeatureDeconvolution.

    Args:
        feature_map: pyopenms FeatureMap containing detected features.
        adducts: list of adduct strings or OpenMS potential_adducts strings.
            Supported bracket forms: "[M+H]+", "[M+Na]+", "[M+K]+", "[M+NH4]+".
            OpenMS form examples: "H:+:0.4", "Na:+:0.25".
        max_charge: maximum charge to consider.
        ppm_tolerance: approximate mass tolerance used for matching (ppm).
            OpenMS MetaboliteFeatureDeconvolution uses a global Da tolerance; we approximate it at the
            median m/z of the input features.
        keep_only_backbone: if True, return only the representative (backbone) features.

    Returns:
        De-adducted pyopenms FeatureMap with neutral masses (charge set to 0).
    """
    reference_mz = _reference_mz(feature_map)
    negative_mode = _infer_negative_mode(feature_map, adducts)
    feature_map_in = _feature_map_with_charge(feature_map, negative_mode)

    annotated = oms.FeatureMap()
    deconvolver = _configured_metabolite_deconvolver(
        adducts=adducts,
        max_charge=max_charge,
        negative_mode=negative_mode,
        ppm_tolerance=ppm_tolerance,
        reference_mz=reference_mz,
    )
    deconvolver.compute(
        feature_map_in,
        annotated,
        oms.ConsensusMap(),
        oms.ConsensusMap(),
    )

    return _neutral_feature_map(annotated, keep_only_backbone)


def _reference_mz(feature_map) -> float:
    mzs = []
    for feature in feature_map:
        try:
            mzs.append(float(feature.getMZ()))
        except Exception:
            continue
    return float(np.median(mzs)) if mzs else 100.0


def _infer_negative_mode(feature_map, adducts: List[str] | None) -> bool:
    charges = []
    for feature in feature_map:
        try:
            charges.append(int(feature.getCharge()))
        except Exception:
            continue
    nonzero = [charge for charge in charges if charge != 0]
    if nonzero:
        has_pos = any(charge > 0 for charge in nonzero)
        has_neg = any(charge < 0 for charge in nonzero)
        if has_pos and has_neg:
            raise ValueError("Mixed positive/negative charges in FeatureMap are not supported.")
        return has_neg

    inferred = _charge_signs_from_adducts(adducts)
    if inferred == {"-"}:
        return True
    if inferred == {"+"} or not inferred:
        return False
    raise ValueError(f"Unable to infer polarity from adducts: mixed charge signs {sorted(inferred)}")


def _charge_signs_from_adducts(adducts: List[str] | None) -> set[str]:
    if not adducts:
        return set()

    inferred = set()
    for value in adducts:
        value = value.strip()
        if not value:
            continue
        if value.startswith("[") and value.endswith("]-"):
            inferred.add("-")
        elif value.startswith("[") and value.endswith("]+"):
            inferred.add("+")
        else:
            parts = value.split(":")
            if len(parts) == 3:
                inferred.add(parts[1])
    return inferred


def _feature_map_with_charge(feature_map, negative_mode: bool):
    feature_map_in = oms.FeatureMap()
    for feature in feature_map:
        new_feature = oms.Feature(feature)
        if new_feature.getCharge() == 0:
            new_feature.setCharge(-1 if negative_mode else 1)
        feature_map_in.push_back(new_feature)
    return feature_map_in


def _configured_metabolite_deconvolver(
    adducts: List[str] | None,
    max_charge: int,
    negative_mode: bool,
    ppm_tolerance: float,
    reference_mz: float,
):
    deconvolver = oms.MetaboliteFeatureDeconvolution()
    params = deconvolver.getParameters()
    if params.exists(b"unit"):
        params.setValue(b"unit", b"Da")
    if params.exists(b"negative_mode"):
        params.setValue(b"negative_mode", b"true" if negative_mode else b"false")
    if negative_mode:
        params.setValue(b"charge_min", -int(max_charge))
        params.setValue(b"charge_max", -1)
    else:
        params.setValue(b"charge_min", 1)
        params.setValue(b"charge_max", int(max_charge))
    params.setValue(b"mass_max_diff", float(max(ppm_tolerance * reference_mz / 1e6, 0.002)))
    if adducts:
        params.setValue(b"potential_adducts", _to_openms_potential_adducts(adducts))
    deconvolver.setParameters(params)
    return deconvolver


def _to_openms_potential_adducts(values: List[str]) -> List[bytes]:
    parsed = [_parse_openms_adduct(value) for value in values if value.strip()]
    parsed = _normalise_adduct_probabilities(parsed)
    return [f"{name}:{charge}:{prob:.6g}".encode() for name, charge, prob in parsed]


def _parse_openms_adduct(value: str) -> tuple[str, str, float]:
    value = value.strip()
    if value.startswith("["):
        if value.startswith("[M+") and value.endswith("]+"):
            name = value[len("[M+") : -len("]+")]
            return name, "+", 1.0
        if value == "[M-H]-":
            return "H-1", "-", 1.0
        if value == "[M+Cl]-":
            return "Cl", "-", 1.0
        raise ValueError(f"Unsupported adduct string '{value}' for pyopenms")

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported OpenMS potential_adduct string '{value}'")
    name, charge, prob = parts
    return name, charge, float(prob)


def _normalise_adduct_probabilities(parsed: List[tuple[str, str, float]]):
    normalised = list(parsed)
    for charge_sign in ("+", "-"):
        indices = [i for i, (_, charge, _) in enumerate(normalised) if charge == charge_sign]
        if not indices:
            continue
        total = sum(normalised[i][2] for i in indices)
        if total <= 0:
            raise ValueError(
                f"Invalid OpenMS potential_adduct probabilities for charge '{charge_sign}'"
            )
        for i in indices:
            name, charge, prob = normalised[i]
            normalised[i] = (name, charge, prob / total)
    return normalised


def _neutral_feature_map(annotated, keep_only_backbone: bool):
    neutral = oms.FeatureMap()
    for feature in annotated:
        if keep_only_backbone and not _is_backbone_feature(feature):
            continue

        neutral_feature = oms.Feature()
        neutral_feature.setMZ(_neutral_mass(feature))
        neutral_feature.setRT(float(feature.getRT()))
        neutral_feature.setIntensity(float(feature.getIntensity()))
        neutral_feature.setCharge(0)
        _copy_group_meta(feature, neutral_feature)
        neutral.push_back(neutral_feature)
    return neutral


def _is_backbone_feature(feature) -> bool:
    try:
        return int(feature.getMetaValue(b"is_backbone")) == 1
    except Exception:
        return True


def _neutral_mass(feature) -> float:
    charge = int(feature.getCharge())
    if charge == 0:
        charge = 1

    neutral_mass = float(feature.getMZ()) * abs(charge)
    try:
        adduct_mass = float(feature.getMetaValue(b"dc_charge_adduct_mass"))
    except Exception:
        return neutral_mass

    # OpenMS reports adduct masses using atomic masses (neutral species), while
    # MS m/z shifts correspond to charged species (atomic +/- electron mass).
    return neutral_mass - adduct_mass + ELECTRON_MASS * charge


def _copy_group_meta(source, target):
    try:
        target.setMetaValue(b"Group", source.getMetaValue(b"Group"))
    except Exception:
        pass
