from numba import njit, float64, float32, int32, types
from typing import List, Tuple, Optional
import numpy as np

from vimms.Common import CHROM_TYPE_EMPIRICAL


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
def bisect_right(a: List[float64], x: float64) -> int32:
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
def get_distance(rts: List[float64], query_rt: float64) -> float64:
    which_above = bisect_right(rts, query_rt)
    which_below = which_above - 1
    rt_below = rts[which_below]
    rt_above = rts[which_above]
    return (query_rt - rt_below) / (rt_above - rt_below)


@njit
def rt_match(min_rt: float64, max_rt: float64, query_rt: float64) -> float64:
    return min_rt < query_rt < max_rt


@njit
def interpolate(value_above: float64, value_below: float64, rts: List[float64],
                query_rt: float64) -> float64:
    return value_below + (value_above - value_below) * get_distance(rts, query_rt)


@njit
def get_relative_value(query_rt: float64, min_rt: float64, max_rt: float64, rts: List[float64],
                       intensities: List[float64]) -> float64:
    if not rt_match(min_rt, max_rt, query_rt):
        return None
    which_above = bisect_right(rts, query_rt)
    which_below = which_above - 1
    intensity_below = intensities[which_below]
    intensity_above = intensities[which_above]
    value = interpolate(intensity_above, intensity_below, rts, query_rt)
    return value


@njit
def isolation_match(number: float64, ranges: List[Tuple[float64, float64]]) -> bool:
    joined_ranges = np.column_stack((ranges[:, 0], ranges[:, 1]))
    within_range = np.any(
        np.logical_and(number >= joined_ranges[:, 0], number <= joined_ranges[:, 1]))
    return within_range


@njit
def get_mz_value(mz: float64, mul: float64, add: float64, mz_to_add: float64) -> float64:
    transformed_mz = adduct_transformation(mz, mul, add)
    return transformed_mz + mz_to_add


@njit
def get_relative_mz(chrom_type: types.unicode_type, query_rt: float64, chrom_min_rt: float64,
                    chrom_max_rt: float64, chrom_rts: List[float64],
                    chrom_mzs: List[float32]) -> float64:
    if chrom_type == CHROM_TYPE_EMPIRICAL:
        return get_relative_value(query_rt, chrom_min_rt, chrom_max_rt, chrom_rts, chrom_mzs)
    return 0.0


@njit
def get_mz_ms1(mz: float64, mul: float64, add: float64, chrom_type: types.unicode_type,
               query_rt: float64, chemical_rt: float64,
               chrom_min_rt: float64, chrom_max_rt: float64,
               chrom_rts: List[float64], chrom_mzs: List[float32]) -> float64:
    relative_mz = get_relative_mz(chrom_type, query_rt - chemical_rt,
                                  chrom_min_rt, chrom_max_rt,
                                  chrom_rts, chrom_mzs)
    return get_mz_value(mz, mul, add, relative_mz)
