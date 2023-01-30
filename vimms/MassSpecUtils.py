from collections import defaultdict

import numpy as np
from numba import njit, float64, int32, types
from numba_stats import norm

from vimms.Common import CHROM_TYPE_EMPIRICAL, CHROM_TYPE_CONSTANT, CHROM_TYPE_FUNCTIONAL, \
    ADDUCT_TERMS, CHEM_RT_IDX, CHROM_MZ_IDX, CHROM_REL_INTENSITY_IDX, ISOTOPE_MZ_IDX, MUL_IDX, \
    ADD_IDX, ISOTOPE_PROP_IDX, ADDUCT_PROB_IDX, CHEM_MAX_INTENSITY_IDX, \
    WHICH_ISOTOPES_IDX, WHICH_ADDUCTS_IDX


@njit
def adduct_transformation(mz: float64, mul: float64, add: float64) -> float64:
    """
    Transform m/z value according to the selected adduct transformation.

    Args:
        mz: the m/z value to check
        mul: adduct multiplier term
        add: adduct addition term

    Returns: the new adduct-transformed m/z value

    """
    transformed = (mz * mul) + add
    return transformed


@njit
def bisect_right(a: np.ndarray, x: float64) -> int32:
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Copied from https://raw.githubusercontent.com/python/cpython/3.11/Lib/bisect.py
    so we can jit this in ViMMS for performance reason
    """

    lo = 0
    hi = None
    key = None

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo


@njit
def rt_match(min_rt: float64, max_rt: float64, query_rt: float64) -> float64:
    return min_rt < query_rt < max_rt


@njit
def interpolate(which_above: int32, which_below: int32,
                value_above: float64, value_below: float64,
                rts: np.ndarray, query_rt: float64) -> float64:
    rt_below = rts[which_below]
    rt_above = rts[which_above]
    distance = (query_rt - rt_below) / (rt_above - rt_below)
    return value_below + (value_above - value_below) * distance


@njit
def get_relative_value(query_rt: float64, min_rt: float64, max_rt: float64, rts: np.ndarray,
                       values: np.ndarray) -> float64:
    if not rt_match(min_rt, max_rt, query_rt):
        return None
    which_above = bisect_right(rts, query_rt)
    which_below = which_above - 1
    value_below = values[which_below]
    value_above = values[which_above]
    value = interpolate(which_above, which_below, value_above, value_below, rts, query_rt)
    return value


@njit
def get_relative_mz_intensity_values(query_rt: float64,
                                     mz_arr: np.ndarray, rt_arr: np.ndarray,
                                     intensity_arr: np.ndarray,
                                     with_intensity: bool) -> float64:
    which_above = bisect_right(rt_arr, query_rt)
    which_below = which_above - 1
    rt_below = rt_arr[which_below]
    rt_above = rt_arr[which_above]
    distance = (query_rt - rt_below) / (rt_above - rt_below)

    value_below = mz_arr[which_below]
    value_above = mz_arr[which_above]
    rel_mz = value_below + (value_above - value_below) * distance

    rel_int = np.nan
    if with_intensity:
        value_below = intensity_arr[which_below]
        value_above = intensity_arr[which_above]
        rel_int = value_below + (value_above - value_below) * distance

    return rel_mz, rel_int


@njit
def get_mz_value(mz: float64, mul: float64, add: float64, mz_to_add: float64) -> float64:
    transformed_mz = adduct_transformation(mz, mul, add)
    return transformed_mz + mz_to_add


@njit
def get_relative_mz(chrom_type: types.unicode_type, query_rt: float64, chrom_min_rt: float64,
                    chrom_max_rt: float64, chrom_rts: np.ndarray,
                    chrom_mzs: np.ndarray) -> float64:
    if chrom_type == CHROM_TYPE_EMPIRICAL:
        return get_relative_value(query_rt, chrom_min_rt, chrom_max_rt, chrom_rts, chrom_mzs)
    return 0.0


# @njit
def get_relative_intensity_functional_normal(query_rt, cutoff, parameters):
    x = np.array([cutoff / 2])
    loc, scale = parameters
    ppf = norm._ppf(x, loc, scale)[0]
    rv = np.exp((-0.5 * (query_rt + ppf - loc) ** 2) / scale ** 2)
    return rv


def generate_chem_ms1_peaks_for_ms1(chems, scan_time, cdc):
    return generate_chem_ms1_peaks(chems, scan_time, cdc, True)


def generate_chem_ms1_peaks_for_ms2(chems, scan_time, cdc):
    return generate_chem_ms1_peaks(chems, scan_time, cdc, False)


def generate_chem_ms1_peaks(chems, scan_time, cdc, with_intensity):
    all_chems, all_data, all_chrom_mz_arr, all_chrom_rt_arr, all_chrom_intensity_arr, \
    chrom_shape_cutoff, chrom_shape_parameters, chrom_type, distribution_name = \
        cdc.collect_chem_data(chems)

    assert len(all_chems) == len(all_data)
    assert len(all_chems) == len(all_chrom_mz_arr)
    assert len(all_chems) == len(all_chrom_rt_arr)
    assert len(all_chems) == len(all_chrom_intensity_arr)

    row_count = len(all_chems)
    peaks = np.empty(row_count)
    which_isotopes = np.empty(row_count)
    which_adducts = np.empty(row_count)

    if row_count > 0:
        data = all_data[:, ISOTOPE_MZ_IDX:CHROM_MZ_IDX + 1]
        peaks = calculate_chem_ms1_peaks(
            data, all_chrom_mz_arr, all_chrom_rt_arr, all_chrom_intensity_arr,
            chrom_shape_cutoff, chrom_shape_parameters, chrom_type,
            distribution_name, scan_time, with_intensity)

        which_isotopes = all_data[:, WHICH_ISOTOPES_IDX].astype(int)
        which_adducts = all_data[:, WHICH_ADDUCTS_IDX].astype(int)

    assert row_count == len(peaks)
    return all_chems, which_isotopes, which_adducts, peaks


class ChemDataCollector():

    def __init__(self, ionisation_mode):
        self.ionisation_mode = ionisation_mode
        self.seen_chems = {}

    def collect_chem_data(self, chems):
        chrom_type = None
        distribution_name = None
        chrom_shape_cutoff = None
        chrom_shape_parameters = (None, None)

        all_chems_data = []
        all_chrom_mz_arr = []
        all_chrom_rt_arr = []
        all_chrom_intensity_arr = []
        all_chems = []

        if len(chems) > 0:
            # Check for the first chem. Assume it's the same for all the others
            chrom = chems[0].chromatogram
            chrom_type = chrom.get_chrom_type()
            if chrom_type == CHROM_TYPE_FUNCTIONAL:
                distribution_name = chrom.distribution_name
                chrom_shape_cutoff = chrom.cutoff
                chrom_shape_parameters = chrom.parameters

            # collect all the relevant information for all chems
            for i in range(len(chems)):
                chemical = chems[i]
                chem_results = self._query_single(chemical)
                all_chems_data.extend(chem_results[0])
                all_chrom_mz_arr.extend(chem_results[1])
                all_chrom_rt_arr.extend(chem_results[2])
                all_chrom_intensity_arr.extend(chem_results[3])
                all_chems.extend(chem_results[4])

        all_chems_data = _get_all_data_arr(all_chems_data)  # FIXME: slow!!

        return all_chems, all_chems_data, all_chrom_mz_arr, all_chrom_rt_arr, \
               all_chrom_intensity_arr, chrom_shape_cutoff, chrom_shape_parameters, \
               chrom_type, distribution_name

    def _query_single(self, chem):
        if chem in self.seen_chems:
            return self.seen_chems[chem]
        else:

            chrom = chem.chromatogram
            chrom_type = chrom.get_chrom_type()
            if chrom_type == CHROM_TYPE_EMPIRICAL:
                chrom_relative_intensity = None
                chrom_mz_arr = chrom.mzs
                chrom_rt_arr = chrom.rts
                chrom_intensity_arr = chrom.intensities
                chrom_mz = None

            elif chrom_type == CHROM_TYPE_FUNCTIONAL:
                chrom_relative_intensity = None
                chrom_mz_arr = None
                chrom_rt_arr = None
                chrom_intensity_arr = None
                chrom_mz = chrom.mz

            elif chrom_type == CHROM_TYPE_CONSTANT:
                chrom_relative_intensity = chrom.relative_intensity
                chrom_mz_arr = None
                chrom_rt_arr = None
                chrom_intensity_arr = None
                chrom_mz = chrom.mz

            single_chem_data = []
            single_chrom_mz_arr = []
            single_chrom_rt_arr = []
            single_chrom_intensity_arr = []
            single_chems = []
            for which_isotope in range(len(chem.isotopes)):
                isotope_mz, isotope_prop, isotope_name = chem.isotopes[which_isotope]

                for which_adduct in range(len(chem.adducts[self.ionisation_mode])):
                    adduct_name, adduct_prob = chem.adducts[self.ionisation_mode][
                        which_adduct]
                    mul, add = ADDUCT_TERMS[adduct_name]

                    row = [
                        isotope_mz,
                        isotope_prop,
                        adduct_prob,
                        mul,
                        add,
                        chem.max_intensity,
                        chem.rt,
                        chrom_relative_intensity,
                        chrom_mz,
                        which_isotope,
                        which_adduct
                    ]
                    single_chem_data.append(row)
                    single_chrom_mz_arr.append(chrom_mz_arr)
                    single_chrom_rt_arr.append(chrom_rt_arr)
                    single_chrom_intensity_arr.append(chrom_intensity_arr)
                    single_chems.append(chem)

            chem_results = (single_chem_data, single_chrom_mz_arr, single_chrom_rt_arr,
                            single_chrom_intensity_arr, single_chems)
            self.seen_chems[chem] = chem_results
            return chem_results


def _get_all_data_arr(all_data):
    all_data = np.array(all_data, dtype=np.float64)
    return all_data


def calculate_chem_ms1_peaks(data,
                             all_chrom_mz_arr,
                             all_chrom_rt_arr,
                             all_chrom_intensity_arr,
                             chrom_shape_cutoff,
                             chrom_shape_parameters,
                             chrom_type,
                             distribution_name,
                             scan_time,
                             with_intensity):
    row_count = len(data)
    rel_mzs = np.zeros(row_count)
    rel_ints = np.zeros(row_count)
    query_rts = scan_time - data[:, CHEM_RT_IDX]

    if chrom_type == CHROM_TYPE_CONSTANT:
        rel_mzs = data[:, CHROM_MZ_IDX]
        rel_ints = data[:, CHROM_REL_INTENSITY_IDX]

    elif chrom_type == CHROM_TYPE_EMPIRICAL:

        # TODO: vectorise this
        for i in range(row_count):
            query_rt = query_rts[i]
            mz_arr = all_chrom_mz_arr[i]
            rt_arr = all_chrom_rt_arr[i]
            intensity_arr = all_chrom_intensity_arr[i]

            rel_mz, rel_int = get_relative_mz_intensity_values(
                query_rt, mz_arr, rt_arr, intensity_arr, with_intensity)
            rel_mzs[i] = rel_mz
            rel_ints[i] = rel_int

    elif chrom_type == CHROM_TYPE_FUNCTIONAL:
        if distribution_name == 'normal':
            rel_mzs = data[:, CHROM_MZ_IDX]
            rel_ints = get_relative_intensity_functional_normal(
                query_rts, chrom_shape_cutoff, chrom_shape_parameters)

        else:
            # TODO: add support for gamma, uniform
            raise NotImplementedError()

    # vectorised m/z and intensity calculation
    peaks = get_scan_mzs_intensities(data, rel_mzs, rel_ints, row_count, with_intensity)
    return peaks


def get_scan_mzs_intensities(data, rel_mzs, rel_ints, row_count, with_intensity):
    transformed_mz = (data[:, ISOTOPE_MZ_IDX] * data[:, MUL_IDX]) + data[:, ADD_IDX]
    scan_mzs = transformed_mz + rel_mzs

    peaks = np.empty((row_count, 5))
    peaks.fill(np.nan)
    peaks[:, 0] = scan_mzs

    if with_intensity:
        intensity = data[:, ISOTOPE_PROP_IDX] * data[:, ADDUCT_PROB_IDX] * \
                    data[:, CHEM_MAX_INTENSITY_IDX]
        scan_intensities = intensity * rel_ints
        peaks[:, 1] = scan_intensities

    return peaks


@njit
def get_mz_ms1(mz: float64, mul: float64, add: float64, chrom_type: types.unicode_type,
               query_rt: float64, chemical_rt: float64,
               chrom_min_rt: float64, chrom_max_rt: float64,
               chrom_rts: np.ndarray, chrom_mzs: np.ndarray) -> float64:
    relative_mz = get_relative_mz(chrom_type, query_rt - chemical_rt,
                                  chrom_min_rt, chrom_max_rt,
                                  chrom_rts, chrom_mzs)
    return get_mz_value(mz, mul, add, relative_mz)


# @njit
def get_mz_msn(mz: float64, mul: float64, add: float64, ms1_parent_isotopes: np.ndarray,
               which_isotope: int32) -> float64:
    isotope_transformation = ms1_parent_isotopes[which_isotope][0] - \
                             ms1_parent_isotopes[0][0]
    mz_value = get_mz_value(mz, mul, add, isotope_transformation)
    return mz_value
