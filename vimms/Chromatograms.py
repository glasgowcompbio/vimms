from abc import ABC, abstractmethod

import numpy as np
import scipy.stats

from vimms.Common import MAX_POSSIBLE_RT


class Chromatogram(ABC):

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


class EmpiricalChromatogram(Chromatogram):
    """
    Empirical Chromatograms to be used within Chemicals
    """

    def __init__(self, rts, mzs, intensities, single_point_length=0.9):
        self.raw_rts = rts
        self.raw_mzs = mzs
        self.raw_intensities = intensities
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
        self.rts = rts - min(rts)
        self.mzs = mzs - np.mean(
            mzs)  # may want to just set this to 0 and remove from input
        self.intensities = intensities / max(intensities)
        # chromatogramDensityNormalisation(rts, intensities)

        self.min_rt = min(self.rts)
        self.max_rt = max(self.rts)

    def get_apex_rt(self):
        max_pos = 0
        max_intensity = self.intensities[0]
        for i, intensity in enumerate(self.intensities):
            if intensity > max_intensity:
                max_intensity = intensity
                max_pos = i
        return self.rts[max_pos]

    def get_relative_intensity(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            neighbours_which = self._get_rt_neighbours_which(query_rt)
            intensity_below = self.intensities[neighbours_which[0]]
            intensity_above = self.intensities[neighbours_which[1]]
            return intensity_below + (
                    intensity_above - intensity_below) * self._get_distance(
                query_rt)

    def get_relative_mz(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            neighbours_which = self._get_rt_neighbours_which(query_rt)
            mz_below = self.mzs[neighbours_which[0]]
            mz_above = self.mzs[neighbours_which[1]]
            return mz_below + (mz_above - mz_below) * self._get_distance(
                query_rt)

    def _get_rt_neighbours(self, query_rt):
        which_rt_below, which_rt_above = self._get_rt_neighbours_which(
            query_rt)
        rt_below = self.rts[which_rt_below]
        rt_above = self.rts[which_rt_above]
        return [rt_below, rt_above]

    def _get_rt_neighbours_which(self, query_rt):
        # find the max index of self.rts smaller than query_rt
        pos = np.where(self.rts <= query_rt)[0]
        which_rt_below = pos[-1]

        # take the min index of self.rts larger than query_rt
        pos = np.where(self.rts > query_rt)[0]
        which_rt_above = pos[0]
        return [which_rt_below, which_rt_above]

    def _get_distance(self, query_rt):
        rt_below, rt_above = self._get_rt_neighbours(query_rt)
        return (query_rt - rt_below) / (rt_above - rt_below)

    def _rt_match(self, query_rt):
        return self.min_rt < query_rt < self.max_rt

    def __eq__(self, other):
        if not isinstance(other, EmpiricalChromatogram):
            # don't attempt to compare against unrelated types
            return NotImplemented
        res = np.array_equal(sorted(self.raw_mzs), sorted(other.raw_mzs)) and \
            np.array_equal(sorted(self.raw_rts), sorted(other.raw_rts)) and \
            np.array_equal(sorted(self.raw_intensities), sorted(other.raw_intensities))
        return res


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
        self.parameters = parameters
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
        self.max_rt = self.distrib.ppf(
            1 - (self.cutoff / 2)) - self.distrib.ppf(self.cutoff / 2)

    def get_relative_intensity(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        elif self.distribution_name == 'normal':
            rv = np.exp(
                (-0.5 * (query_rt + self.distrib.ppf(self.cutoff / 2) -
                         self.parameters[0]) ** 2) / self.parameters[
                    1] ** 2)
            return rv
        else:
            return (self.distrib.pdf(
                query_rt + self.distrib.ppf(self.cutoff / 2)) * (
                            1 / (1 - self.cutoff)))

    def get_relative_mz(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            return self.mz

    def _rt_match(self, query_rt):
        if query_rt < 0 or query_rt > self.max_rt:
            return False
        else:
            return True

    def get_apex_rt(self):
        if self.distribution_name == 'uniform':
            return (self.max_rt - self.min_rt) / 2
        elif self.distribution_name == 'normal':
            return (self.max_rt - self.min_rt) / 2
        else:
            raise NotImplementedError()
