# test deisotoping and isotope generation

import numpy as np
import pytest

from vimms.Chemicals import Isotopes, Adducts
from vimms.Common import Formula, ADDUCT_TERMS, POSITIVE, PROTON_MASS
from vimms.Deisotoping import Deisotoper
from vimms.MassSpecUtils import adduct_transformation


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
