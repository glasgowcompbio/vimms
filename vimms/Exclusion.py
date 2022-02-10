from abc import abstractmethod

import numpy as np
from intervaltree import IntervalTree
from loguru import logger

from vimms.Common import ScanParameters


###############################################################################
# DEW Exclusions
###############################################################################


class ExclusionItem(object):
    """
    A class to store the item to exclude when computing dynamic
    exclusion window
    """

    def __init__(self, from_mz, to_mz, from_rt, to_rt, frag_at):
        """
        Creates a dynamic exclusion item
        :param from_mz: m/z lower bounding box
        :param to_mz: m/z upper bounding box
        :param from_rt: RT lower bounding box
        :param to_rt: RT upper bounding box
        """
        self.from_mz = from_mz
        self.to_mz = to_mz
        self.from_rt = from_rt
        self.to_rt = to_rt
        self.frag_at = frag_at
        self.mz = (self.from_mz + self.to_mz) / 2.
        self.rt = self.frag_at

    def peak_in(self, mz, rt):
        if self.rt_match(rt) and self.mz_match(mz):
            return True
        else:
            return False

    def rt_match(self, rt):
        if rt >= self.from_rt and rt <= self.to_rt:
            return True
        else:
            return False

    def mz_match(self, mz):
        if mz >= self.from_mz and mz <= self.to_mz:
            return True
        else:
            return False

    def __repr__(self):
        return 'ExclusionItem mz=(%f, %f) rt=(%f-%f)' % (
            self.from_mz, self.to_mz, self.from_rt, self.to_rt)

    def __lt__(self, other):
        if self.from_mz <= other.from_mz:
            return True
        else:
            return False


class BoxHolder(object):
    """
    A class to allow quick lookup of boxes (e.g. exclusion items,
    targets, etc). Creates an interval tree on mz as this is likely to
    narrow things down quicker. Also has a method for returning an rt
    interval tree for a particular mz and an mz interval tree for
    a particular rt
    """

    def __init__(self):
        self.boxes_mz = IntervalTree()
        self.boxes_rt = IntervalTree()

    def add_box(self, box):
        """
        Add a box to the IntervalTree
        """
        mz_from = box.from_mz
        mz_to = box.to_mz
        rt_from = box.from_rt
        rt_to = box.to_rt
        self.boxes_mz.addi(mz_from, mz_to, box)
        self.boxes_rt.addi(rt_from, rt_to, box)

    def check_point(self, mz, rt):
        """
        Find the boxes that match this mz and rt value
        """
        regions = self.boxes_mz.at(mz)
        hits = set()
        for r in regions:
            if r.data.rt_match(rt):
                hits.add(r.data)
        return hits

    # FIXME: this produces a different result from check_point, do not use
    # def check_point_2(self, mz, rt):
    #     """
    #     An alternative method that searches both trees
    #     Might be faster if there are lots of rt ranges that
    #     can map to a particular mz value
    #     """
    #     mz_regions = self.boxes_mz.at(mz)
    #     rt_regions = self.boxes_rt.at(rt)
    #     inter = mz_regions.intersection(rt_regions)
    #     return [r.data for r in inter]

    def is_in_box(self, mz, rt):
        """
        Check if this mz and rt is in *any* box
        """
        hits = self.check_point(mz, rt)
        if len(hits) > 0:
            return True
        else:
            return False

    def is_in_box_mz(self, mz):
        """
        Check if an mz value is in any box
        """
        regions = self.boxes_mz.at(mz)
        if len(regions) > 0:
            return True
        else:
            return False

    def is_in_box_rt(self, rt):
        """
        Check if an rt value is in any box
        """
        regions = self.boxes_rt.at(rt)
        if len(regions) > 0:
            return True
        else:
            return False

    def get_subset_rt(self, rt):
        """
        Create an interval tree based upon mz for all boxes active at rt
        """
        regions = self.boxes_rt.at(rt)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it

    def get_subset_mz(self, mz):
        """
        Create an interval tree based upon rt fro all boxes active at mz
        """
        regions = self.boxes_mz.at(mz)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it


class TopNExclusion(object):
    def __init__(self, initial_exclusion_list=None):
        self.exclusion_list = BoxHolder()
        if initial_exclusion_list is not None:  # add initial list, if provided
            for initial in initial_exclusion_list:
                self.exclusion_list.add_box(initial)

    def is_excluded(self, mz, rt):
        """
        Checks if a pair of (mz, rt) value is currently excluded by
        dynamic exclusion window
        :param mz: m/z value
        :param rt: RT value
        :return: True if excluded (with weight 0.0), False otherwise
         (weight 1.0)
        """
        excluded = self.exclusion_list.is_in_box(mz, rt)
        if excluded:
            logger.debug(
                'Excluded precursor ion mz {:.4f} rt {:.2f}'.format(mz, rt))
            return True, 0.0
        else:
            return False, 1.0

    def update(self, current_scan, ms2_tasks):
        """
        Updates the state of this exclusion object based on the current
        ms1 scan and scheduled ms2 tasks
        :param current_scan: the current MS1 scan
        :param ms2_tasks: scheduled ms2 tasks
        """
        rt = current_scan.rt
        for task in ms2_tasks:
            for precursor in task.get('precursor_mz'):
                mz = precursor.precursor_mz
                mz_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL)
                rt_tol = task.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
                x = self._get_exclusion_item(mz, rt, mz_tol, rt_tol)
                self.exclusion_list.add_box(x)
                logger.debug(
                    'Time {:.6f} Created dynamic temporary exclusion '
                    'window mz ({}-{}) rt ({}-{})'.format(
                        rt, x.from_mz, x.to_mz, x.from_rt, x.to_rt))

    def _get_exclusion_item(self, mz, rt, mz_tol, rt_tol):
        mz_lower = mz * (1 - mz_tol / 1e6)
        mz_upper = mz * (1 + mz_tol / 1e6)
        rt_lower = rt - rt_tol
        # I think this is mostly for topN (iterative) exclusion method
        rt_upper = rt + rt_tol
        x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower,
                          to_rt=rt_upper,
                          frag_at=rt)
        return x


class WeightedDEWExclusion(TopNExclusion):
    def __init__(self, rt_tol, exclusion_t_0):
        super().__init__()
        self.rt_tol = rt_tol
        self.exclusion_t_0 = exclusion_t_0
        assert self.exclusion_t_0 <= self.rt_tol

    def is_excluded(self, mz, rt):
        """
        Checks if a pair of (mz, rt) value is currently excluded
        by the weighted dynamic exclusion window
        :param mz: m/z value
        :param rt: RT value
        :return: True if excluded, False otherwise
        """
        boxes = self.exclusion_list.check_point(mz, rt)
        if len(boxes) > 0:
            # compute weights for all the boxes that contain this (mz, rt)
            weights = []
            for b in boxes:
                _, w = compute_weight(rt, b.frag_at, self.rt_tol,
                                      self.exclusion_t_0)
                weights.append(w)

            # use the min weight -- seems to work well
            w = min(weights)
            flag = False if w == 1.0 else True
            return flag, w

        else:
            return False, 1.0


def compute_weight(current_rt, frag_at, rt_tol, exclusion_t_0):
    if frag_at is None:
        # never been fragmented before, always include (weight 1.0)
        return False, 1.0
    elif current_rt >= frag_at + rt_tol:
        # outside the windows of rt_tol, always include (weight > 1.0)
        return False, 1.0
    elif current_rt <= frag_at + exclusion_t_0:
        # fragmented but within exclusion_t_0, always exclude (weight 0.0)
        return True, 0.0
    else:
        # compute weight according to the WeightedDEW scheme
        weight = (current_rt - (exclusion_t_0 + frag_at)) / (
            rt_tol - exclusion_t_0)
        if weight > 1:
            logger.warning('exclusion weight %f is greater than 1 ('
                           'current_rt %f exclusion_t_0 %f frag_at %f '
                           'rt_tol %f)' % (weight, current_rt,
                                           exclusion_t_0, frag_at, rt_tol))
        # assert weight <= 1, weight
        return True, weight


###############################################################################
# Filters
###############################################################################


class ScoreFilter():
    @abstractmethod
    def filter(self): pass


class MinIntensityFilter(ScoreFilter):
    def __init__(self, min_ms1_intensity):
        self.min_ms1_intensity = min_ms1_intensity

    def filter(self, intensities):
        return np.array(intensities) > self.min_ms1_intensity


class DEWFilter(ScoreFilter):
    def __init__(self, rt_tol):
        self.rt_tol = rt_tol

    def filter(self, current_rt, last_frag_rts):
        # Handles None values by converting to NaN for which all
        # comparisons return 0
        return np.logical_not(
            current_rt - np.array(
                last_frag_rts, dtype=np.double) <= self.rt_tol)


class WeightedDEWFilter(ScoreFilter):
    def __init__(self, exclusion):
        self.exclusion = exclusion

    def filter(self, current_rt, last_frag_rts, rois):
        weights = []
        for roi in rois:
            last_mz, last_rt, last_intensity = roi.get_last_datum()
            is_exc, weight = self.exclusion.is_excluded(last_mz, last_rt)
            weights.append(weight)
        return np.array(weights)


class LengthFilter(ScoreFilter):
    def __init__(self, min_roi_length_for_fragmentation):
        self.min_roi_length_for_fragmentation = \
            min_roi_length_for_fragmentation

    def filter(self, roi_lengths):
        return roi_lengths >= self.min_roi_length_for_fragmentation


class SmartROIFilter(ScoreFilter):
    def filter(self, rois):
        # if this is a normal ROI object, always return True for everything
        # otherwise track the status based on the SmartROI rules
        can_fragments = np.array([roi.get_can_fragment() for roi in rois])
        return can_fragments


if __name__ == '__main__':
    e = ExclusionItem(1.1, 1.2, 3.4, 3.5, 3.45)
    f = ExclusionItem(1.0, 1.4, 3.3, 3.6, 3.45)
    g = ExclusionItem(2.1, 2.2, 3.2, 3.5, 3.45)
    b = BoxHolder()
    b.add_box(e)
    b.add_box(f)
    b.add_box(g)
    print(b.is_in_box(1.15, 3.55))
    print(b.is_in_box(1.15, 3.75))
