"""
This file implements various exclusion and filtering criteria in a modular way that can be reused
elsewhere, e.g. in controllers.
"""
from abc import abstractmethod, ABC

import numpy as np
from intervaltree import IntervalTree
from loguru import logger

from vimms.Common import ScanParameters


###############################################################################
# DEW Exclusions
###############################################################################


class ExclusionItem():
    """
    A class to store the item to exclude when computing dynamic
    exclusion window
    """

    def __init__(self, from_mz, to_mz, from_rt, to_rt, frag_at):
        """
        Creates a dynamic exclusion item

        Args:
            from_mz: m/z lower bounding box
            to_mz: m/z upper bounding box
            from_rt: RT lower bounding box
            to_rt: RT upper bounding box
            frag_at: RT when this ExclusionItem is fragmented
        """
        self.from_mz = from_mz
        self.to_mz = to_mz
        self.from_rt = from_rt
        self.to_rt = to_rt
        self.frag_at = frag_at
        self.mz = (self.from_mz + self.to_mz) / 2.
        self.rt = self.frag_at

    def peak_in(self, mz, rt):
        """
        Checks that a peak described by its (mz, rt) values lies in this box.
        Args:
            mz: the m/z value to check
            rt: the RT value to check

        Returns: True if it's a match, False otherwise.

        """
        if self.rt_match(rt) and self.mz_match(mz):
            return True
        else:
            return False

    def rt_match(self, rt):
        """
        Checks that a certain RT point lies in this box
        Args:
            rt: the RT value to check

        Returns: True if it's a match, False otherwise.

        """
        if rt >= self.from_rt and rt <= self.to_rt:
            return True
        else:
            return False

    def mz_match(self, mz):
        """
        Checks that a certain m/z point lies in this box
        Args:
            mz: the m/z value to check

        Returns: True if it's a match, False otherwise.

        """
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


class BoxHolder():
    """
    A class to allow quick lookup of boxes (e.g. exclusion items,
    targets, etc). Creates an interval tree on mz as this is likely to
    narrow things down quicker. Also has a method for returning an rt
    interval tree for a particular mz and an mz interval tree for
    a particular RT.
    """

    def __init__(self):
        """
        Initialise a BoxHolder object
        """
        self.boxes_mz = IntervalTree()
        self.boxes_rt = IntervalTree()
        
    def __iter__(self):
        return (inv.data for inv in self.boxes_rt.items())

    def add_box(self, box):
        """
        Add a box to the IntervalTree

        Args:
            box: the box to add

        Returns: None

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

        Args:
            mz: the m/z to check
            rt: the RT to check

        Returns: the list of hits (boxes that contain this point)

        """
        regions = self.boxes_mz.at(mz)
        hits = set()
        for r in regions:
            if r.data.rt_match(rt):
                hits.add(r.data)
        return hits

    # FIXME: this produces a different result from check_point, do not use yet
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

        Args:
            mz: the m/z to check
            rt: the RT to check

        Returns: True if it's a match, False otherwise.

        """
        hits = self.check_point(mz, rt)
        if len(hits) > 0:
            return True
        else:
            return False

    def is_in_box_mz(self, mz):
        """
        Check if an mz value is in any box

        Args:
            mz: the m/z to check

        Returns: True if it's a match, False otherwise.

        """
        regions = self.boxes_mz.at(mz)
        if len(regions) > 0:
            return True
        else:
            return False

    def is_in_box_rt(self, rt):
        """
        Check if an RT value is in any box

        Args:
            rt: the m/z to check

        Returns: True if it's a match, False otherwise.

        """
        regions = self.boxes_rt.at(rt)
        if len(regions) > 0:
            return True
        else:
            return False

    def get_subset_rt(self, rt):
        """
        Create an interval tree based upon mz for all boxes active at rt

        Args:
            rt: the RT to check

        Returns: a new BoxHolder object containing the subset of boxes at RT

        """
        regions = self.boxes_rt.at(rt)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it

    def get_subset_mz(self, mz):
        """
        Create an interval tree based upon mz for all boxes active at m/z

        Args:
            mz: the m/z value to check

        Returns: a new BoxHolder object containing the subset of boxes at m/z

        """
        regions = self.boxes_mz.at(mz)
        it = BoxHolder()
        for r in regions:
            box = r.data
            it.add_box(box)
        return it


class TopNExclusion():
    """
    A class that perform standard dynamic exclusion for Top-N.
    This is based on checked whether an m/z and RT value lies in certain exclusion boxes.
    """

    def __init__(self, initial_exclusion_list=None):
        """
        Initialise a Top-N dynamic exclusion object
        Args:
            initial_exclusion_list: the initial list of boxes, if provided
        """
        self.exclusion_list = BoxHolder()
        if initial_exclusion_list is not None:  # add initial list
            for initial in initial_exclusion_list:
                self.exclusion_list.add_box(initial)

    def is_excluded(self, mz, rt):
        """
        Checks if a pair of (mz, rt) value is currently excluded by
        dynamic exclusion window

        Args:
            mz: m/z value
            rt: RT value

        Returns: True if excluded (with weight 0.0), False otherwise (weight 1.0).

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

        Args:
            current_scan: the current MS1 scan
            ms2_tasks: scheduled ms2 tasks

        Returns: None

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
        """
        Create a new [vimms.Exclusion.ExclusionItem][] object based on the (mz, rt) values
        as well as the tolerances.
        Args:
            mz: the m/z value
            rt: the RT value
            mz_tol: the m/z tolerance (in ppm)
            rt_tol: the RT tolerance (in seconds)

        Returns: a new [vimms.Exclusion.ExclusionItem][] object

        """
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
    """
    A class that perform weighted dynamic exclusion for Top-N.
    This is further described in our paper 'Rapid Development ...'
    """

    def __init__(self, rt_tol, exclusion_t_0):
        """
        Initialises a weighted dynamic exclusion object
        Args:
            rt_tol: the RT tolerance (in seconds)
            exclusion_t_0: WeightedDEW parameter
        """
        super().__init__()
        self.rt_tol = rt_tol
        self.exclusion_t_0 = exclusion_t_0
        assert self.exclusion_t_0 <= self.rt_tol

    def is_excluded(self, mz, rt):
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
    """
    Compute the weight for current RT at frag_at given the RT tolerance and other parameters.
    Args:
        current_rt: the current RT value
        frag_at: the retention time when fragmentation last occured
        rt_tol: RT tolerance (in seconds)
        exclusion_t_0: weighted DEW parameter

    Returns: a new weight

    """
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


class ScoreFilter(ABC):
    """
    Base class for various filters
    """
    @abstractmethod
    def filter(self):
        pass


class MinIntensityFilter(ScoreFilter):
    """
    A class that implements minimum intensity filter
    """
    def __init__(self, min_ms1_intensity):
        """
        Initialises the minimum intensity filter
        Args:
            min_ms1_intensity: the minimum intensity to check
        """
        self.min_ms1_intensity = min_ms1_intensity

    def filter(self, intensities):
        """
        Check whether intensity values are above or below the threshold
        Args:
            intensities: an array of intensity values

        Returns: an array of indicators for the filter

        """
        return np.array(intensities) > self.min_ms1_intensity


class DEWFilter(ScoreFilter):
    """
    A class that implements dynamic exclusion filter
    """
    def __init__(self, rt_tol):
        """
        Initialises a dynamic exclusion filter based on time only
        Args:
            rt_tol: the RT tolerance (in seconds)
        """
        self.rt_tol = rt_tol

    def filter(self, current_rt, last_frag_rts):
        """
        Check whether intensity values are above or below the threshold
        Args:
            current_rt: the current RT value
            intensities (array): an array of last fragmented RT values

        Returns: an array of indicators for the filter
        """

        # Handles None values by converting to NaN for which all
        # comparisons return 0
        return np.logical_not(
            current_rt - np.array(
                last_frag_rts, dtype=np.double) <= self.rt_tol)


class WeightedDEWFilter(ScoreFilter):
    """
    A class that implements weighted dynamic exclusion filter
    """
    def __init__(self, exclusion):
        """
        Initialises a weighted dynamic exclusion filter

        Args:
            exclusion: a [vimms.Exclusion.ExclusionItem][] object
        """
        self.exclusion = exclusion

    def filter(self, current_rt, last_frag_rts, rois):
        """
        Check whether ROIs are excluded or not based on weighted dynamic exclusion filter
        Args:
            current_rt: the current RT value
            last_frag_rts: the last fragmented RT values of ROIs
            rois: a list of [vimms.Roi.Roi][] objects.

        Returns: a numpy array of weights for each ROI.

        """
        weights = []
        for roi in rois:
            last_mz, last_rt, last_intensity = roi.get_last_datum()
            is_exc, weight = self.exclusion.is_excluded(last_mz, last_rt)
            weights.append(weight)
        return np.array(weights)


class LengthFilter(ScoreFilter):
    """
    A class that implements a check on minimum length of ROI for fragmentation
    """
    def __init__(self, min_roi_length_for_fragmentation):
        """
        Initialise a length filter

        Args:
            min_roi_length_for_fragmentation: the minimum length of ROI for fragmentation
        """
        self.min_roi_length_for_fragmentation = \
            min_roi_length_for_fragmentation

    def filter(self, roi_lengths):
        """
        Check that ROI lengths are above the threshold
        Args:
            roi_lengths: a numpy array of ROI lengths

        Returns: an array of indicator whether the lengths are above threshold

        """
        return roi_lengths >= self.min_roi_length_for_fragmentation


class SmartROIFilter(ScoreFilter):
    """
    A class that implements SmartROI filtering criteria.
    For more details, refer to our paper 'Rapid Development ...'
    """

    def filter(self, rois):
        """
        Filter ROIs based on SmartROI rules.


        Args:
            rois: a list of [vimms.Roi.Roi] objects. if this is a normal ROI object,
                  always return True for everything otherwise track the status based
                  on the SmartROI rules

        Returns: an array of indicator whether ROI can be fragmented or not.

        """
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
