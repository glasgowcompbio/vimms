from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.stats
from numba_stats import norm

from vimms.Common import MAX_POSSIBLE_RT, CHROM_TYPE_EMPIRICAL, CHROM_TYPE_CONSTANT, \
    CHROM_TYPE_FUNCTIONAL
from vimms.MassSpecUtils import rt_match, get_relative_value, \
    get_relative_intensity_functional_normal


class Chromatogram(metaclass=ABCMeta):
    __slots__ = ()

    @abstractmethod
    def get_relative_intensity(self, query_rt):
        pass

    @abstractmethod
    def get_relative_mz(self, query_rt):
        pass

    @abstractmethod
    def _rt_match(self, rt):
        pass

    @abstractmethod
    def get_apex_rt(self):
        pass

    @abstractmethod
    def get_chrom_type(self):
        pass


class EmpiricalChromatogram(Chromatogram):
    """
    Empirical Chromatograms to be used within Chemicals
    """

    __slots__ = (
        "raw_rts",
        "raw_mzs",
        "raw_intensities",
        "rts",
        "mzs",
        "intensities",
        "min_rt",
        "max_rt",
        "raw_min_rt",
        "raw_max_rt"
    )

    def __init__(self, rts, mzs, intensities, single_point_length=0.9):
        self.raw_rts = rts
        self.raw_mzs = mzs
        self.raw_intensities = intensities

        self.raw_min_rt = min(self.raw_rts)
        self.raw_max_rt = max(self.raw_rts)

        # ensures that all arrays are in sorted order
        if len(rts) > 1:
            p = rts.argsort()
            rts = rts[p]
            mzs = mzs[p]
            intensities = intensities[p]
        else:
            rts = np.array([rts[0] - 0.5 * single_point_length,
                            rts[0] + 0.5 * single_point_length])
            mzs = np.array([mzs[0], mzs[0]])
            intensities = np.array([intensities[0], intensities[0]])
        # normalise arrays
        self.rts = rts - self.raw_min_rt
        self.mzs = mzs - np.mean(mzs)  # may want to just set this to 0 and remove from input
        self.intensities = intensities / max(intensities)
        # chromatogramDensityNormalisation(rts, intensities)

        self.min_rt = np.min(self.rts)
        self.max_rt = np.max(self.rts)

    def get_apex_rt(self):
        max_pos = 0
        max_intensity = self.intensities[0]
        for i, intensity in enumerate(self.intensities):
            if intensity > max_intensity:
                max_intensity = intensity
                max_pos = i
        return self.rts[max_pos]

    def get_relative_intensity(self, query_rt):
        return get_relative_value(query_rt, self.min_rt, self.max_rt, self.rts, self.intensities)

    def get_relative_mz(self, query_rt):
        return get_relative_value(query_rt, self.min_rt, self.max_rt, self.rts, self.mzs)

    def _rt_match(self, query_rt):
        return rt_match(self.min_rt, self.max_rt, query_rt)

    def __eq__(self, other):
        if not isinstance(other, EmpiricalChromatogram):
            # don't attempt to compare against unrelated types
            return NotImplemented
        res = np.array_equal(sorted(self.raw_mzs), sorted(other.raw_mzs)) and \
              np.array_equal(sorted(self.raw_rts), sorted(other.raw_rts)) and \
              np.array_equal(sorted(self.raw_intensities), sorted(other.raw_intensities))
        return res

    def get_chrom_type(self):
        return CHROM_TYPE_EMPIRICAL


class ConstantChromatogram(Chromatogram):
    def __init__(self):
        self.mz = 0.0
        self.relative_intensity = 1.0
        self.min_rt = 0.0
        self.max_rt = MAX_POSSIBLE_RT

    def get_relative_intensity(self, query_rt):
        return self.relative_intensity

    def get_relative_mz(self, query_rt):
        return self.mz

    def _rt_match(self, query_rt):
        return True

    def get_apex_rt(self):
        return self.min_rt

    def get_chrom_type(self):
        return CHROM_TYPE_CONSTANT


# Make this more generalisable. Make scipy.stats... as input,
# However this makes it difficult to do the cutoff
class FunctionalChromatogram(Chromatogram):
    """
    Functional Chromatograms to be used within Chemicals
    """

    def __init__(self, distribution, parameters, cutoff=0.01):
        self.cutoff = cutoff
        self.mz = 0
        self.distribution_name = distribution
        self.parameters = np.array(parameters, dtype=float)
        if distribution == "normal":
            self.distrib = scipy.stats.norm(parameters[0], parameters[1])
        elif distribution == "gamma":
            self.distrib = scipy.stats.gamma(parameters[0], parameters[1],
                                             parameters[2])
        elif distribution == "uniform":
            self.distrib = scipy.stats.uniform(parameters[0], parameters[1])
        else:
            raise NotImplementedError("distribution not implemented")

        self.min_rt = 0

        if self.distribution_name == 'normal':
            loc, scale = self.parameters
            x1 = np.array([1 - (self.cutoff / 2)])
            x2 = np.array([self.cutoff / 2])
            self.max_rt = (norm._ppf(x1, loc, scale) - norm._ppf(x2, loc, scale))[0]
        else:
            self.max_rt = (
                    self.distrib.ppf(1 - (self.cutoff / 2)) - self.distrib.ppf(self.cutoff / 2)
            )

    def get_relative_intensity(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        elif self.distribution_name == 'normal':
            return get_relative_intensity_functional_normal(
                query_rt, self.cutoff, self.parameters)
        else:
            return (self.distrib.pdf(query_rt + self.distrib.ppf(self.cutoff / 2)) * (
                        1 / (1 - self.cutoff)))


    def get_relative_mz(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            return self.mz

    def _rt_match(self, query_rt):
        return rt_match(self.min_rt, self.max_rt, query_rt)

    def get_apex_rt(self):
        if self.distribution_name == 'uniform':
            return (self.max_rt - self.min_rt) / 2
        elif self.distribution_name == 'normal':
            return (self.max_rt - self.min_rt) / 2
        else:
            raise NotImplementedError()

    def get_chrom_type(self):
        return CHROM_TYPE_FUNCTIONAL
