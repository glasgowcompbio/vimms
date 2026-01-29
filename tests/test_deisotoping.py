# test deisotoping and isotope generation

import numpy as np

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
