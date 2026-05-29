# test deisotoping and isotope generation

import numpy as np
import pytest

from vimms.Chemicals import Isotopes, Adducts
from vimms.Common import (
    C13_MZ_DIFF,
    Formula,
    ADDUCT_NAMES_POS,
    ADDUCT_PRIOR_POS,
    ADDUCT_TERMS,
    POSITIVE,
    PROTON_MASS,
)
from vimms.Deisotoping import Deisotoper
from vimms.MassSpecUtils import adduct_transformation


def _fmap_to_merged_peaks(feature_map):
    # Consolidate near-duplicate neutral masses emitted by OpenMS MFD.
    # We use an absolute tolerance rather than ppm because these duplicates
    # are typically micro-Da artifacts (e.g. atomic-vs-ion mass handling).
    abs_tol = 1e-3

    peaks = sorted(
        ((float(feature.getMZ()), float(feature.getIntensity())) for feature in feature_map),
        key=lambda x: x[0],
    )
    if not peaks:
        return []

    merged: list[tuple[float, float]] = []
    current_mz, current_intensity = peaks[0]
    for mz, intensity in peaks[1:]:
        if abs(mz - current_mz) <= abs_tol:
            current_intensity += intensity
            continue
        merged.append((current_mz, current_intensity))
        current_mz, current_intensity = mz, intensity
    merged.append((current_mz, current_intensity))

    return merged


def test_isotope_distribution_multi_element():
    formula = Formula("C10H16N2O2S")
    isotopes = Isotopes(formula)
    peaks = isotopes.get_isotopes(total_proportion=0.99)

    proportions = [peak[1] for peak in peaks]
    mzs = [peak[0] for peak in peaks]

    assert len(peaks) > 1
    assert np.isclose(sum(proportions), 1.0, atol=1e-6)
    assert all(mzs[i] < mzs[i + 1] for i in range(len(mzs) - 1))


def test_isotope_distribution_chlorine_m2_peak():
    formula = Formula("C5H10Cl2")
    isotopes = Isotopes(formula)
    peaks = isotopes.get_isotopes(total_proportion=0.99)

    mono_mz = peaks[0][0]
    deltas = [mz - mono_mz for mz, _, _ in peaks[1:]]

    assert any(np.isclose(delta, 1.997, atol=0.01) for delta in deltas)


def test_adduct_terms_chloride_uses_chloride_anion_mass():
    # [M+Cl]- uses Cl- (atomic mass + electron mass), not neutral Cl.
    from vimms.Common import ELECTRON_MASS, NATURAL_ISOTOPES

    _, cl_shift = ADDUCT_TERMS["M+Cl"]
    cl_atomic = NATURAL_ISOTOPES["Cl"][0][0]
    assert np.isclose(cl_shift - cl_atomic, ELECTRON_MASS, atol=1e-12)


def test_default_positive_adducts_exclude_unsupported_dimers():
    assert "2M+H" not in ADDUCT_NAMES_POS
    assert "2M+NH4" not in ADDUCT_NAMES_POS
    assert "2M+H" not in ADDUCT_PRIOR_POS
    assert "2M+NH4" not in ADDUCT_PRIOR_POS
    assert "2M+H" not in ADDUCT_TERMS
    assert "2M+NH4" not in ADDUCT_TERMS


def test_unsupported_dimer_adducts_raise_clear_error():
    formula = Formula("C10H20")

    with pytest.raises(ValueError, match="multimer isotope envelopes"):
        Adducts(formula, adduct_prior_dict={POSITIVE: {"2M+H": 1.0}})

    with pytest.raises(ValueError, match="multimer isotope envelopes"):
        Adducts(formula, adduct_prior_dict={POSITIVE: {"2M+NH4": 1.0}})


def test_potassium_adduct_uses_corrected_name():
    assert "M+2K-H" in ADDUCT_NAMES_POS
    assert "M+2K-H" in ADDUCT_PRIOR_POS
    assert ADDUCT_TERMS["M+2K-H"] == (1, 76.919040)
    assert "M+2K+H" not in ADDUCT_NAMES_POS
    assert "M+2K+H" not in ADDUCT_PRIOR_POS
    assert "M+2K+H" not in ADDUCT_TERMS

    formula = Formula("C10H20")
    with pytest.raises(ValueError, match="use 'M\\+2K-H' instead"):
        Adducts(formula, adduct_prior_dict={POSITIVE: {"M+2K+H": 1.0}})


def test_isotope_distribution_keeps_prominent_halogen_high_mass_peaks():
    chlorinated = Isotopes(Formula("C30H40Cl2O5S2")).get_isotopes(total_proportion=0.99)
    chlorinated_deltas = [mz - chlorinated[0][0] for mz, _, _ in chlorinated[1:]]

    brominated = Isotopes(Formula("C60H100Br2O10")).get_isotopes(total_proportion=0.99)
    brominated_deltas = [mz - brominated[0][0] for mz, _, _ in brominated[1:]]

    assert len(chlorinated) > 20
    assert any(np.isclose(delta, 3.994, atol=0.02) for delta in chlorinated_deltas)
    assert len(brominated) > 20
    assert any(np.isclose(delta, 6.003, atol=0.02) for delta in brominated_deltas)


def test_isotope_distribution_warns_when_explicit_peak_cap_truncates():
    formula = Formula("C30H40Cl2O5S2")
    isotopes = Isotopes(formula)

    with pytest.warns(RuntimeWarning, match="max_peaks prevented"):
        peaks = isotopes.get_isotopes(total_proportion=0.99, max_peaks=20)

    assert len(peaks) == 20


def test_isotope_distribution_preserves_mono_when_filtered():
    formula = Formula("C500H1000")
    isotopes = Isotopes(formula)
    peaks = isotopes.get_isotopes(total_proportion=0.99, min_prob=0.01)

    assert np.isclose(peaks[0][0], formula.mass, atol=1e-6)


def test_deisotoper_recovers_mono():
    formula = Formula("C10H16N2O2S")
    isotopes = Isotopes(formula)
    adducts = Adducts(formula, adduct_prior_dict={POSITIVE: {"M+H": 1.0}})
    adduct_name = adducts.get_adducts()[POSITIVE][0][0]
    mul, add = ADDUCT_TERMS[adduct_name]

    peaks = []
    for mz, proportion, _ in isotopes.get_isotopes(total_proportion=0.99):
        adducted_mz = adduct_transformation(mz, mul, add)
        peaks.append((adducted_mz, proportion * 1e5))

    deisotoper = Deisotoper(ppm_tolerance=10.0, max_charge=1, min_isotopes=2)
    clusters = deisotoper.deisotope(peaks)

    assert len(clusters) == 1
    expected_mz = formula.mass + PROTON_MASS
    assert np.isclose(clusters[0].monoisotopic_mz, expected_mz, atol=1e-3)


def test_deisotoper_handles_m_plus_2_only():
    peaks = [(100.0, 1e5), (101.997, 6e4)]
    deisotoper = Deisotoper(ppm_tolerance=10.0, max_charge=1, min_isotopes=2)
    clusters = deisotoper.deisotope(peaks)

    assert len(clusters) == 1
    assert np.isclose(clusters[0].monoisotopic_mz, 100.0, atol=1e-6)


def test_pyopenms_deisotope_helper():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.003355, 5e4), (102.00671, 2e4)]
    _, mzs, intensities = deisotope_with_pyopenms(peaks, min_isopeaks=2, max_isopeaks=3)

    assert len(mzs) <= len(peaks)
    assert len(mzs) == len(intensities)


def test_pyopenms_deadduct_helper():
    pytest.importorskip("pyopenms")
    import pyopenms as oms

    from vimms.Deisotoping import deadduct_with_pyopenms

    fmap = oms.FeatureMap()
    feature = oms.Feature()
    feature.setMZ(100.0)
    feature.setIntensity(1e5)
    fmap.push_back(feature)

    output = deadduct_with_pyopenms(fmap, adducts=["[M+H]+"])
    assert output.size() >= 1


def test_pyopenms_deadduct_multiple_features():
    pytest.importorskip("pyopenms")
    import pyopenms as oms

    from vimms.Deisotoping import deadduct_with_pyopenms

    fmap = oms.FeatureMap()
    for mz, intensity in [(100.0, 1e5), (122.989218, 4e4)]:
        feature = oms.Feature()
        feature.setMZ(mz)
        feature.setIntensity(intensity)
        fmap.push_back(feature)

    output = deadduct_with_pyopenms(fmap, adducts=["[M+H]+", "[M+Na]+"])
    assert output.size() >= 1


def test_pyopenms_deisotope_empty_peaks():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    spectrum, mzs, intensities = deisotope_with_pyopenms([])
    assert spectrum is None
    assert mzs.size == 0
    assert intensities.size == 0


def test_pyopenms_deisotope_sorts_input():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(102.00671, 2e4), (100.0, 1e5), (101.003355, 5e4)]
    _, mzs, intensities = deisotope_with_pyopenms(peaks, min_isopeaks=2, max_isopeaks=3)

    assert mzs.size == intensities.size
    assert all(mzs[i] <= mzs[i + 1] for i in range(len(mzs) - 1))


def test_pyopenms_deisotope_keeps_non_isotopic_peaks_by_default():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5)]
    _, mzs, intensities = deisotope_with_pyopenms(peaks)

    assert mzs.size == 1
    assert intensities.size == 1
    assert np.isclose(mzs[0], 100.0, atol=1e-12)


def test_pyopenms_deisotope_keep_only_deisotoped_returns_mono_peak():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.003355, 5e4), (102.00671, 2e4)]
    _, mzs, intensities = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=3,
        max_isopeaks=3,
        make_single_charged=False,
    )

    assert mzs.size == 1
    assert intensities.size == 1
    assert np.isclose(mzs[0], 100.0, atol=1e-6)
    assert np.isclose(intensities[0], 1e5, atol=1e-6)


def test_pyopenms_deisotope_add_up_intensity_sums_cluster():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.003355, 5e4), (102.00671, 2e4)]
    _, mzs, intensities = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=3,
        max_isopeaks=3,
        make_single_charged=False,
        add_up_intensity=True,
    )

    assert mzs.size == 1
    assert np.isclose(intensities[0], 1e5 + 5e4 + 2e4, atol=1e-6)


def test_pyopenms_deisotope_annotation_arrays_present_and_consistent():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.003355, 5e4), (102.00671, 2e4)]
    spectrum, mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=3,
        max_isopeaks=3,
        make_single_charged=False,
        annotate_charge=True,
        annotate_iso_peak_count=True,
        annotate_features=True,
    )

    assert spectrum is not None
    assert mzs.size == spectrum.size()

    integer_arrays = spectrum.getIntegerDataArrays()
    names = {arr.getName() for arr in integer_arrays}
    assert {"charge", "iso_peak_count", "feature_number"} <= names
    assert all(len(arr) == spectrum.size() for arr in integer_arrays)

    charge_arr = next(arr for arr in integer_arrays if arr.getName() == "charge")
    iso_count_arr = next(arr for arr in integer_arrays if arr.getName() == "iso_peak_count")
    feature_arr = next(arr for arr in integer_arrays if arr.getName() == "feature_number")

    assert int(charge_arr[0]) == 1
    assert int(iso_count_arr[0]) == 3
    assert int(feature_arr[0]) == 0


def test_pyopenms_deisotope_make_single_charged_converts_charge_two():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    mz0 = 100.0
    c13_diff = 1.003355
    peaks = [(mz0, 1e5), (mz0 + c13_diff / 2, 5e4), (mz0 + 2 * c13_diff / 2, 2e4)]
    spectrum, mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=2,
        max_charge=2,
        min_isopeaks=3,
        max_isopeaks=3,
        make_single_charged=True,
        annotate_charge=True,
    )

    assert spectrum is not None
    assert mzs.size == 1
    assert np.isclose((2 * mz0) - mzs[0], PROTON_MASS, atol=1e-4)

    charge_arr = next(arr for arr in spectrum.getIntegerDataArrays() if arr.getName() == "charge")
    assert int(charge_arr[0]) == 2


def test_pyopenms_deisotope_start_intensity_check_controls_strictness():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.003355, 1.5e5)]
    _, strict_mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_isopeaks=2,
        max_isopeaks=2,
        use_decreasing_model=True,
        start_intensity_check=1,
    )
    _, relaxed_mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_isopeaks=2,
        max_isopeaks=2,
        use_decreasing_model=True,
        start_intensity_check=2,
    )

    assert strict_mzs.size == 0
    assert relaxed_mzs.size == 1
    assert np.isclose(relaxed_mzs[0], 100.0, atol=1e-6)


def test_pyopenms_deisotope_is_c13_only_drops_m_plus_2_only():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    peaks = [(100.0, 1e5), (101.997, 6e4)]
    _, mzs, intensities = deisotope_with_pyopenms(
        peaks, keep_only_deisotoped=True, min_isopeaks=2, max_isopeaks=2
    )

    assert mzs.size == 0
    assert intensities.size == 0


def test_pyopenms_deisotope_is_c13_only_on_multi_element_generator():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    formula = Formula("Cl2")
    isotopes = Isotopes(formula)
    isotope_peaks = isotopes.get_isotopes(total_proportion=0.99)
    peaks = [(mz, proportion * 1e5) for mz, proportion, _ in isotope_peaks]

    mono_mz = isotope_peaks[0][0]
    deltas = [mz - mono_mz for mz, _, _ in isotope_peaks[1:]]
    assert any(np.isclose(delta, 1.997, atol=0.01) for delta in deltas)
    assert not any(np.isclose(delta, 1.003355, atol=0.01) for delta in deltas)

    homegrown_clusters = Deisotoper(ppm_tolerance=10.0, max_charge=1, min_isotopes=2).deisotope(
        peaks
    )
    assert len(homegrown_clusters) == 1

    _, mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=2,
        max_isopeaks=10,
    )
    assert mzs.size == 0


def test_pyopenms_deisotope_recovers_mono_from_generated_adducted_isotopes():
    pytest.importorskip("pyopenms")
    from vimms.Deisotoping import deisotope_with_pyopenms

    formula = Formula("C10H16N2O2S")
    isotopes = Isotopes(formula)
    adducts = Adducts(formula, adduct_prior_dict={POSITIVE: {"M+H": 1.0}})
    adduct_name = adducts.get_adducts()[POSITIVE][0][0]
    mul, add = ADDUCT_TERMS[adduct_name]

    peaks = []
    for mz, proportion, _ in isotopes.get_isotopes(total_proportion=0.99):
        adducted_mz = adduct_transformation(mz, mul, add)
        peaks.append((adducted_mz, proportion * 1e5))

    _, mzs, _ = deisotope_with_pyopenms(
        peaks,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=2,
        max_isopeaks=10,
        make_single_charged=False,
    )

    expected_mz = formula.mass + PROTON_MASS
    assert mzs.size >= 1
    assert np.any(np.isclose(mzs, expected_mz, atol=1e-3))
    assert np.isclose(mzs.min(), expected_mz, atol=1e-3)


def test_end_to_end_deadduct_then_homegrown_deisotope_recovers_neutral_mono():
    pytest.importorskip("pyopenms")
    import pyopenms as oms

    from vimms.Deisotoping import deadduct_with_pyopenms

    formula = Formula("C10H16N2O2S")
    isotopes = Isotopes(formula)
    isotope_peaks = isotopes.get_isotopes(total_proportion=0.99)[:8]

    fmap = oms.FeatureMap()
    uid = 1
    for mz, proportion, _ in isotope_peaks:
        for adduct_name, weight in (("M+H", 1.0), ("M+Na", 0.6)):
            mul, add = ADDUCT_TERMS[adduct_name]
            feature = oms.Feature()
            feature.setMZ(adduct_transformation(mz, mul, add))
            feature.setIntensity(float(proportion) * 1e5 * weight)
            feature.setRT(0.0)
            feature.setCharge(1)
            feature.setUniqueId(uid)
            uid += 1
            fmap.push_back(feature)

    neutral = deadduct_with_pyopenms(
        fmap,
        adducts=["[M+H]+", "[M+Na]+"],
        ppm_tolerance=10.0,
        keep_only_backbone=False,
    )
    assert neutral.size() > 0

    merged = _fmap_to_merged_peaks(neutral)
    assert len(merged) <= len(isotope_peaks)

    deisotoper = Deisotoper(ppm_tolerance=10.0, max_charge=1, min_isotopes=2)
    clusters = deisotoper.deisotope(merged)

    assert any(np.isclose(c.monoisotopic_mz, formula.mass, atol=1e-2) for c in clusters)


def test_end_to_end_deadduct_then_pyopenms_deisotope_recovers_neutral_mono():
    pytest.importorskip("pyopenms")
    import pyopenms as oms

    from vimms.Deisotoping import deadduct_with_pyopenms, deisotope_with_pyopenms

    neutral_mz = 200.0
    isotope_peaks = [
        (neutral_mz, 1e5),
        (neutral_mz + C13_MZ_DIFF, 5e4),
        (neutral_mz + 2 * C13_MZ_DIFF, 2e4),
    ]

    fmap = oms.FeatureMap()
    uid = 1
    for mz, intensity in isotope_peaks:
        for adduct_name, weight in (("M+H", 1.0), ("M+Na", 0.5)):
            mul, add = ADDUCT_TERMS[adduct_name]
            feature = oms.Feature()
            feature.setMZ(adduct_transformation(mz, mul, add))
            feature.setIntensity(float(intensity) * weight)
            feature.setRT(0.0)
            feature.setCharge(1)
            feature.setUniqueId(uid)
            uid += 1
            fmap.push_back(feature)

    neutral = deadduct_with_pyopenms(
        fmap,
        adducts=["[M+H]+", "[M+Na]+"],
        ppm_tolerance=10.0,
        keep_only_backbone=False,
    )
    merged = _fmap_to_merged_peaks(neutral)

    _, mzs, _ = deisotope_with_pyopenms(
        merged,
        keep_only_deisotoped=True,
        min_charge=1,
        max_charge=1,
        min_isopeaks=3,
        max_isopeaks=3,
        make_single_charged=False,
    )
    assert mzs.size == 1
    assert np.isclose(mzs[0], neutral_mz, atol=1e-3)


def test_end_to_end_deadduct_negative_then_homegrown_deisotope_recovers_neutral_mono():
    pytest.importorskip("pyopenms")
    import pyopenms as oms

    from vimms.Deisotoping import deadduct_with_pyopenms

    cl_ion_mass = 34.969402  # [M+Cl]- shift in observed m/z

    formula = Formula("C10H16N2O2S")
    isotopes = Isotopes(formula)
    isotope_peaks = isotopes.get_isotopes(total_proportion=0.99)[:6]

    fmap = oms.FeatureMap()
    uid = 1
    for mz, proportion, _ in isotope_peaks:
        for shift, weight in ((-PROTON_MASS, 1.0), (cl_ion_mass, 0.5)):
            feature = oms.Feature()
            feature.setMZ(float(mz + shift))
            feature.setIntensity(float(proportion) * 1e5 * weight)
            feature.setRT(0.0)
            feature.setCharge(-1)
            feature.setUniqueId(uid)
            uid += 1
            fmap.push_back(feature)

    neutral = deadduct_with_pyopenms(
        fmap,
        adducts=["[M-H]-", "[M+Cl]-"],
        ppm_tolerance=10.0,
        max_charge=1,
        keep_only_backbone=False,
    )
    assert neutral.size() > 0

    merged = _fmap_to_merged_peaks(neutral)
    deisotoper = Deisotoper(ppm_tolerance=10.0, max_charge=1, min_isotopes=2)
    clusters = deisotoper.deisotope(merged)

    assert any(np.isclose(c.monoisotopic_mz, formula.mass, atol=1e-2) for c in clusters)
