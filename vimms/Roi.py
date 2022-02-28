"""
Provides implementation of Regions-of-Interest (ROI) objects that are used for
real-time ROI tracking in various controller. Additionally, ROIs can also be
loaded from an mzML file and converted into Chemical objects for simulation
input.
"""
import copy
import bisect
import sys
import itertools
from collections import Counter
from statistics import mean

import numpy as np
import pandas as pd
import pylab as plt
import pymzml
import statsmodels.api as sm
from loguru import logger
from mass_spec_utils.data_import.mzmine import (
    load_picked_boxes, map_boxes_to_scans
)
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.alignment import Peak, PeakSet
from scipy.stats import pearsonr

from vimms.Box import GenericBox
from vimms.Chromatograms import EmpiricalChromatogram
from vimms.Common import (
    MZ_UNITS_PPM, MZ_UNITS_DA, 
    ROI_TYPE_NORMAL, ROI_TYPE_SMART,
    path_or_mzml
)
from vimms.Evaluation import (
    load_peakonly_boxes, load_xcms_boxes, 
    get_precursor_intensities
)
from vimms.MassSpec import Scan


class Roi():
    """
    A class to store an ROI (Regions-of-interest). An ROI is a region of
    consecutive scans that potentially form a chromatographic peak. This is the
    first step in peak detection but before the region is identified to be a
    peak or not. This class maintains 3 lists -- mz, rt and intensity.
    When a new point (mz,rt,intensity) is added, it updates the list and the
    mean mz which is required.
    """

    def __init__(self, mz, rt, intensity, id=None):
        """
        Constructs a new ROI
        :param mz: the initial m/z of this ROI. Can be either a single value
        or a list of values.
        :param rt: the initial rt  of this ROI. Can be either a single value
        or a list of values.
        :param intensity: the initial intensity  of this ROI. Can be either
        a single value or a list of values.
        """
        self.id = id
        self.fragmentation_events = []
        self.fragmentation_intensities = []
        self.max_fragmentation_intensity = 0.0
        self.mz_list = self._to_list(mz)
        self.rt_list = self._to_list(rt)
        self.intensity_list = self._to_list(intensity)
        self.n = len(self.mz_list)
        self.mz_sum = sum(self.mz_list)
        self.length_in_seconds = self.rt_list[-1] - self.rt_list[0]
        self.is_fragmented = False
        self.can_fragment = True
        
        self.min_rt, self.max_rt = min(self.rt_list), max(self.rt_list)
        self.min_mz, self.max_mz = min(self.mz_list), max(self.mz_list)
        self.min_intensity, self.max_intensity = min(self.intensity_list), max(self.intensity_list)

    def fragmented(self):
        """
        Sets flags to indicate that this ROI can or has been fragmented
        """
        self.is_fragmented = True
        self.can_fragment = True

    def get_mean_mz(self):
        """
        Returns the mean m/z values of points in this ROI
        """
        return self.mz_sum / self.n

    def get_max_intensity(self):
        """
        Returns the maximum intensity value of this ROI
        """
        return self.max_intensity

    def get_min_intensity(self):
        """
        Returns the minimum intensity value of this ROI
        """
        return self.min_intensity

    def get_autocorrelation(self, lag=1):
        """
        Computes auto-correlation of this ROI intensity signal
        """
        return pd.Series(self.intensity_list).autocorr(lag=lag)

    def estimate_apex(self):
        """
        Returns the apex retention time
        """
        return self.rt_list[np.argmax(self.intensity_list)]

    def add(self, mz, rt, intensity):
        """
        Adds a point to this ROI
        :param mz: the m/z value to add
        :param rt: the retention time value to add
        :param intensity: the intensity value to add
        """
        self.mz_list.append(mz)
        self.rt_list.append(rt)
        self.intensity_list.append(intensity)
        self.mz_sum += mz
        self.n += 1
        self.length_in_seconds = self.rt_list[-1] - self.rt_list[0]
        self.min_rt, self.max_rt = min(self.min_rt, rt), max(self.max_rt, rt)
        self.min_mz, self.max_mz = min(self.min_mz, mz), max(self.max_mz, mz)
        self.min_intensity, self.max_intensity = min(self.min_intensity, intensity), max(self.max_intensity, intensity)

    def add_fragmentation_event(self, scan, precursor_intensity):
        """
        Stores the fragmentation events (MS2 scan) linked to this ROI
        :param scan: the MS2 scan
        :param precursor_intensity: the precursor intensity
        """
        self.fragmentation_events.append(scan)
        self.fragmentation_intensities.append(precursor_intensity)
        self.max_fragmentation_intensity = max(self.fragmentation_intensities)

    def to_chromatogram(self):
        """
        Converts this ROI to a ViMMS EmpiricalChromatogram object
        """
        if self.n == 0:
            return None
        chrom = EmpiricalChromatogram(np.array(self.rt_list),
                                      np.array(self.mz_list),
                                      np.array(self.intensity_list))
        return chrom

    def to_box(self, min_rt_width, min_mz_width, rt_shift=0, mz_shift=0):
        """
        Returns a generic box representation of this ROI
        :param min_rt_width: minimum RT width of the box
        :param min_mz_width: minimum m/z width of the box
        :param rt_shift: shift in retention time, if any
        :param mz_shift: shift in m/z, if any
        """
        return GenericBox(
                    self.min_rt + rt_shift, 
                    self.max_rt + rt_shift, 
                    self.min_mz + mz_shift,
                    self.max_mz + mz_shift,
                    min_xwidth=min_rt_width, 
                    min_ywidth=min_mz_width, 
                    intensity=self.max_fragmentation_intensity, 
                    id=self.id,
                    roi=self
              )

    def get_boxes_overlap(self, boxes, min_rt_width, min_mz_width, rt_shift=0, mz_shift=0):
        """
        TODO: ask Ross to add comment
        Returns ???
        :param boxes: the boxes to check
        :param min_rt_width: minimum RT width of the box
        :param min_mz_width: minimum m/z width of the box
        :param rt_shift: shift in retention time, if any
        :param mz_shift: shift in m/z, if any
        """

        roi_box = self.to_box(min_rt_width, min_mz_width, rt_shift, mz_shift)
        # print(roi_box)
        overlaps = [roi_box.overlap_2(box) for box in boxes]
        return overlaps

    def get_roi_overlap(self, boxes, min_rt_width, min_mz_width, rt_shift=0,
                        mz_shift=0):
        """
        TODO: ask Ross to add comment
        Returns ???
        :param boxes: the boxes to check
        :param min_rt_width: minimum RT width of the box
        :param min_mz_width: minimum m/z width of the box
        :param rt_shift: shift in retention time, if any
        :param mz_shift: shift in m/z, if any
        """
        roi_box = self.to_box(min_rt_width, min_mz_width, rt_shift, mz_shift)
        overlaps = [roi_box.overlap_3(box) for box in boxes]
        return overlaps

    def get_last_datum(self):
        """
        Returns the last (m/z, rt, intensity) point of this ROI
        """
        return self.mz_list[-1], self.rt_list[-1], self.intensity_list[-1]

    def _to_list(self, val):
        """
        Ensures that the value passed in is a list
        :param val: the value to check, can be either a list or a single value
        """
        values = val if type(val) == list else [val]
        return values

    def __getitem__(self, idx):
        """
        Returns a single (rt, m/z, intensity) point in this ROI at the
        specified index
        :param idx: the index of item to retrieve
        """
        return list(zip(self.rt_list, self.mz_list, self.intensity_list))[idx]

    def __lt__(self, other):
        """
        Compares this ROI to other based on the mean m/z value.
        Used for sorting.
        """
        return self.get_mean_mz() <= other.get_mean_mz()

    def __repr__(self):
        """
        Returns a string representation of this ROI
        """
        return 'ROI with data points=%d fragmentations=%d mz ' \
               '(%.4f-%.4f) rt (%.4f-%.4f)' % (
                   self.n,
                   len(self.fragmentation_events),
                   self.mz_list[0], self.mz_list[-1],
                   self.rt_list[0], self.rt_list[-1])


class SmartRoi(Roi):
    """
    A smarter ROI class that can track the states in which it should be
    fragmented.

    SmartROI is described further in the following paper:
    - Davies, Vinny, et al. "Rapid Development of Improved Data-Dependent
    Acquisition Strategies." Analytical chemistry 93.14 (2021): 5676-5683.
    """
    INITIAL_WAITING = 0
    CAN_FRAGMENT = 1
    AFTER_FRAGMENT = 2
    POST_PEAK = 3

    def __init__(self, mz, rt, intensity, initial_length_seconds=5,
                 reset_length_seconds=100,
                 intensity_increase_factor=2, dew=15, drop_perc=0.01, id=None):
        """
        Constructs a new Smart ROI
        :param mz: the initial m/z of this ROI.
        Can be either a single value or a list of values.
        :param rt: the initial rt  of this ROI.
        Can be either a single value or a list of values.
        :param intensity: the initial intensity  of this ROI.
        Can be either a single value or a list of values.
        :param initial_length_seconds: the initial length (in seconds) before
        this ROI can be fragmented
        :param reset_length_seconds: the length (in seconds) before this ROI
        can be fragmented again (CAN_FRAGMENT)
        :param intensity_increase_factor: a factor of which the intensity
        should increase from the **minimum** since
        last fragmentation before this ROI can be fragmented again
        :param drop_perc: percentage drop in intensity since last fragmentation
         before this ROI can be fragmented again
        :param id: the ID of this ROI
        """
        super().__init__(mz, rt, intensity, id=id)

        if initial_length_seconds > 0:
            self.status = SmartRoi.INITIAL_WAITING
            self.set_can_fragment(False)
        else:
            self.status = SmartRoi.CAN_FRAGMENT
            self.set_can_fragment(True)

        self.min_frag_intensity = None
        self.intensity_diff = 0

        self.initial_length_seconds = initial_length_seconds
        self.reset_length_seconds = reset_length_seconds
        self.intensity_increase_factor = intensity_increase_factor
        self.drop_perc = drop_perc
        self.dew = dew

    def fragmented(self):
        """
        Sets this SmartROI as having been fragmented
        """
        self.is_fragmented = True
        self.set_can_fragment(False)
        self.fragmented_index = len(self.mz_list) - 1
        self.status = SmartRoi.AFTER_FRAGMENT

    def get_status(self):
        """
        Returns the current status of this SmartROI
        """
        if self.status == 0:
            return "INITIAL_WAITING"
        elif self.status == 1:
            return "CAN_FRAGMENT"
        elif self.status == 2:
            return "AFTER_FRAGMENT"
        elif self.status == 3:
            return "POST_PEAK"

    # flake8: noqa: C901
    def add(self, mz, rt, intensity):
        """
        Adds a point to this SmartROI
        :param mz: the m/z value to add
        :param rt: the retention time value to add
        :param intensity: the intensity value to add
        """
        super().add(mz, rt, intensity)
        if self.status == SmartRoi.INITIAL_WAITING:
            if self.length_in_seconds >= self.initial_length_seconds:
                self.status = SmartRoi.CAN_FRAGMENT
                self.set_can_fragment(True)
        elif self.status == SmartRoi.AFTER_FRAGMENT:
            # in a period after a fragmentation has happened
            # if enough time has elapsed, reset everything
            if self.rt_list[-1] - self.rt_list[
                self.fragmented_index] > self.reset_length_seconds:
                self.status = SmartRoi.CAN_FRAGMENT
                self.set_can_fragment(True)
            elif self.rt_list[-1] - self.rt_list[
                self.fragmented_index] > self.dew:
                # standard DEW has expired
                # find the min intensity since the frag
                # check current intensity -- if it is 5* when we fragmented,
                # we can go again
                min_since_frag = min(
                    self.intensity_list[self.fragmented_index:])
                if self.intensity_list[
                    -1] > min_since_frag * self.intensity_increase_factor:
                    self.status = SmartRoi.CAN_FRAGMENT
                    self.set_can_fragment(True)
                elif self.intensity_list[-1] < self.drop_perc * \
                        self.intensity_list[self.fragmented_index]:
                    # signal has dropped, but ROI still exists.
                    self.status = SmartRoi.CAN_FRAGMENT
                    self.set_can_fragment(True)

        # code below never happens
        elif self.status == SmartRoi.POST_PEAK:
            if self.rt_list[-1] - self.rt_list[
                self.fragmented_index] > self.dew:
                if self.intensity_list[-1] > self.min_frag_intensity:
                    self.status = SmartRoi.CAN_FRAGMENT
                    self.set_can_fragment(True)

    def get_can_fragment(self):
        """
        Returns the status of whether this SmartROI can be fragmented
        """
        return self.can_fragment

    def set_can_fragment(self, status):
        """
        Sets the status of this SmartROI
        :param status: True if this SmartROI can be fragmented again,
        False otherwise
        """
        self.can_fragment = status
        try:
            self.intensity_diff = abs(
                self.intensity_list[-1] - self.intensity_list[
                    self.fragmented_index])
        except AttributeError:  # no fragmented index
            self.intensity_diff = 0


class RoiParams():
    """
    A parameter object that stores various settings required for ROIBuilder
    """

    def __init__(self, mz_tol=10, min_length=1,
                 min_intensity=0, min_roi_intensity_for_fragmentation=0,
                 start_rt=0, stop_rt=10000000,
                 mz_units=MZ_UNITS_PPM, length_units="scans"):
        """
        Initialises an RoiParams object
        :param mz_tol: m/z  tolerance
        :param min_length: minimum ROI length
        :param min_intensity: minimum intensity to be included for ROI building
        :param start_rt: start RT of scans to be included for ROI building
        :param start_rt: end RT of scans to be included for ROI building
        :param min_roi_intensity_for_fragmentation: keep only the ROIs that can be
        fragmented above this threshold.
        """
        self.mz_tol = mz_tol
        self.min_length = min_length
        self.min_intensity = min_intensity
        self.min_roi_intensity_for_fragmentation = min_roi_intensity_for_fragmentation
        self.start_rt = start_rt
        self.stop_rt = stop_rt
        self.mz_units = mz_units
        self.length_units = length_units

    def __repr__(self):
        return str(self.__dict__)


class RoiBuilder():
    """
    A class to construct ROIs. This can be used in real-time to track ROIs
    in a controller, or for extracting ROIs from an mzML file.
    """

    def __init__(self, mz_tol, rt_tol, min_roi_intensity, min_roi_length,
                 mz_units=MZ_UNITS_PPM,
                 initial_length_seconds=5, reset_length_seconds=100,
                 intensity_increase_factor=2, drop_perc=0.01,
                 length_units="scans",
                 roi_type=ROI_TYPE_NORMAL,
                 min_roi_intensity_for_fragmentation=0):
        """
        Initialises an ROI Builder object.
        :param mz_tol: the m/z tolerance when matching a new point to
        existing ROIs
        :param rt_tol: dynamic exclusion window (in seconds) before an ROI
        can be fragmented again
        :param min_roi_length: minimum length of ROI (in length_units) before
        an ROI is considered to be junk ROI (and can be discarded)
        :param mz_units: m/z units, either in 'Da' or 'ppm'
        :param initial_length_seconds: SmartROI parameter
        :param reset_length_seconds: SmartROI parameter
        :param intensity_increase_factor: SmartROI parameter
        :param drop_perc: SmartROI parameter
        :param length_units: the length unit (either in 'scans' or 'seconds')
        :param roi_type: the type of ROI object generated, either
        ROI_TYPE_NORMAL or ROI_TYPE_SMART
        :param min_roi_intensity_for_fragmentation: keep only the ROIs that can be
        fragmented above this threshold.
        """

        if (mz_units == MZ_UNITS_DA and mz_tol > 1) or (
                mz_units == MZ_UNITS_PPM and mz_tol < 0.1):
            logger.warning(
                f'Is your m/z tolerance unit correct? mz_tol={mz_tol}, '
                f'mz_units={mz_units}')

        # ROI stuff
        self.min_roi_intensity = min_roi_intensity
        self.min_roi_intensity_for_fragmentation = min_roi_intensity_for_fragmentation
        self.mz_tol = mz_tol
        self.mz_units = mz_units
        self.rt_tol = rt_tol
        self.min_roi_length = min_roi_length
        self.length_units = length_units
        self.roi_type = roi_type
        assert self.mz_units in [MZ_UNITS_PPM, MZ_UNITS_DA]
        assert self.roi_type in [ROI_TYPE_NORMAL, ROI_TYPE_SMART]

        # Create ROI
        self.live_roi = []
        self.dead_roi = []
        self.junk_roi = []

        # FIXME: not sure if we actually need the two properties below?
        self.live_roi_fragmented = []
        self.live_roi_last_rt = []  # last fragmentation time of ROI

        # fragmentation to Roi dictionaries
        self.frag_roi_dicts = []  # scan_id, roi_id, precursor_intensity
        self.roi_id_counter = 0

        # Only used by SmartROI
        self.initial_length_seconds = initial_length_seconds
        self.reset_length_seconds = reset_length_seconds
        self.intensity_increase_factor = intensity_increase_factor
        self.drop_perc = drop_perc

    # flake8: noqa: C901
    def update_roi(self, new_scan):
        """
        Updates ROI in real-time based on incoming scans
        :param new_scan: a newly arriving Scan object
        """
        if new_scan.ms_level == 1:

            # Sort ROIs in live_roi according to their m/z values.
            # Ensure that the live roi fragmented flags and the last RT
            # are also consistent with the sorting order.
            order = np.argsort(self.live_roi)
            self.live_roi.sort()
            self.live_roi_fragmented = np.array(self.live_roi_fragmented)[
                order].tolist()
            self.live_roi_last_rt = np.array(self.live_roi_last_rt)[
                order].tolist()

            # Current scan retention time of the MS1 scan is the RT of all
            # points in this scan
            current_ms1_scan_rt = new_scan.rt

            # The set of ROIs that are not grown yet.
            # Initially all currently live ROIs are included, and they're
            # removed once grown.
            not_grew = set(self.live_roi)

            # For every (mz, intensity) in scan ..
            for idx in range(len(new_scan.intensities)):
                intensity = new_scan.intensities[idx]
                mz = new_scan.mzs[idx]

                if intensity >= self.min_roi_intensity:

                    # Create a dummy ROI object to represent the current m/z
                    # value. This produces either a normal ROI or smart ROI
                    # object, depending on self.roi_type
                    roi = self._get_roi_obj(mz, 0, 0, None)

                    # Match dummy ROI to currently live ROIs based on mean
                    # m/z values. If no match, then return None
                    match_roi = match(roi, self.live_roi, self.mz_tol,
                                      mz_units=self.mz_units)
                    if match_roi:

                        # Got a match, so we grow this ROI
                        match_roi.add(mz, current_ms1_scan_rt, intensity)
                        if match_roi in not_grew:
                            not_grew.remove(match_roi)

                    else:

                        # No match, so create a new ROI and insert it in the
                        # right place in the sorted list
                        new_roi = self._get_roi_obj(mz, current_ms1_scan_rt,
                                                    intensity,
                                                    self.roi_id_counter)
                        self.roi_id_counter += 1
                        bisect.insort_right(self.live_roi, new_roi)

                        # Set the fragmented flag of a new ROI to False.
                        # Also set its last fragmented time to None
                        self.live_roi_fragmented.insert(
                            self.live_roi.index(new_roi), False)
                        self.live_roi_last_rt.insert(
                            self.live_roi.index(new_roi), None)

            # Separate the ROIs that have not been grown into dead or junk ROIs
            # Dead ROIs are longer than self.min_roi_length but they haven't
            # been grown. Junk ROIs are too short and not grown.
            for roi in not_grew:

                if self.length_units == "scans":  # number of scans
                    if roi.n >= self.min_roi_length:
                        self.dead_roi.append(roi)
                    else:
                        self.junk_roi.append(roi)

                else:  # seconds
                    if roi.length_in_seconds >= self.min_roi_length:
                        self.dead_roi.append(roi)
                    else:
                        self.junk_roi.append(roi)

                # Remove not-grown ROI from the list of live ROIs
                pos = self.live_roi.index(roi)
                del self.live_roi[pos]
                del self.live_roi_fragmented[pos]
                del self.live_roi_last_rt[pos]

            self.current_roi_ids = [roi.id for roi in self.live_roi]
            self.current_roi_mzs = [roi.mz_list[-1] for roi in self.live_roi]
            self.current_roi_intensities = [roi.intensity_list[-1] for roi in
                                            self.live_roi]

            # FIXME: only the 'scans' mode seems to work on the real mass
            # spec (IAPI), why??
            if self.length_units == "scans":
                self.current_roi_length = np.array(
                    [roi.n for roi in self.live_roi])
            else:
                self.current_roi_length = np.array(
                    [roi.length_in_seconds for roi in self.live_roi])

    def _get_roi_obj(self, mz, rt, intensity, roi_id):
        """
        Constructs a new ROI object based on the currently defined type in
        this builder
        :param mz: the m/z value
        :param rt: the RT value
        :param intensity: the intensity value
        :param roi_id: the ROI id
        """
        if self.roi_type == ROI_TYPE_NORMAL:
            roi = Roi(mz, rt, intensity, id=roi_id)
        elif self.roi_type == ROI_TYPE_SMART:
            roi = SmartRoi(mz, rt, intensity,
                           initial_length_seconds=self.initial_length_seconds,
                           reset_length_seconds=self.reset_length_seconds,
                           intensity_increase_factor=self.intensity_increase_factor,  # noqa
                           dew=self.rt_tol, drop_perc=self.drop_perc,
                           id=roi_id
                           )
        return roi

    def get_mz_intensity(self, i):
        """
        Returns the (m/z, intensity, ROI ID) value of point at position i in
        this ROI
        :param the index of point to return
        """
        mz = self.current_roi_mzs[i]
        intensity = self.current_roi_intensities[i]
        roi_id = self.current_roi_ids[i]
        return mz, intensity, roi_id

    def set_fragmented(self, current_task_id, i, roi_id, rt, intensity):
        """
        Updates this ROI to indicate that it has been fragmented
        """
        # updated fragmented list and times
        self.live_roi_fragmented[i] = True
        self.live_roi_last_rt[i] = rt
        self.live_roi[i].fragmented()

        # Add information on which scan has fragmented this ROI
        self.frag_roi_dicts.append(
            {'scan_id': current_task_id, 'roi_id': roi_id,
             'precursor_intensity': intensity})

        #need to track for intensity non-overlap
        self.live_roi[i].max_fragmentation_intensity = max(
            self.live_roi[i].max_fragmentation_intensity, intensity
        )

    def add_scan_to_roi(self, scan):
        """
        Stores the information on which scans and frag events are associated
        to this ROI
        """
        frag_event_ids = np.array(
            [event['scan_id'] for event in self.frag_roi_dicts])
        which_event = np.where(frag_event_ids == scan.scan_id)[0]
        live_roi_ids = np.array([roi.id for roi in self.live_roi])
        which_roi = \
            np.where(
                live_roi_ids == self.frag_roi_dicts[which_event[0]]['roi_id'])[
                0]
        if len(which_roi) > 0:
            self.live_roi[which_roi[0]].add_fragmentation_event(
                scan,
                self.frag_roi_dicts[which_event[0]]['precursor_intensity'])
            del self.frag_roi_dicts[which_event[0]]
        else:
            pass  # hopefully shouldnt happen

    def get_rois(self):
        """
        Returns all ROIs
        """
        return self.live_roi + self.dead_roi

    def get_good_rois(self):
        """
        Returns all ROIs above filtering criteria
        """
        # length check
        filtered_roi = [roi for roi in self.live_roi if
                        roi.n >= self.min_roi_length]

        # intensity check:
        # Keep only the ROIs that can be fragmented above
        # min_roi_intensity_for_fragmentation threshold.
        all_roi = filtered_roi + self.dead_roi
        if self.min_roi_intensity_for_fragmentation > 0:
            keep = []
            for roi in all_roi:
                if np.count_nonzero(np.array(
                        roi.intensity_list) > self.min_roi_intensity_for_fragmentation) > 0:
                    keep.append(roi)
        else:
            keep = all_roi
        return keep


class RoiAligner():
    """
    A class that aligns multiple ROIs in different samples
    """

    def __init__(self, mz_tolerance_absolute=0.01,
                 mz_tolerance_ppm=10,
                 rt_tolerance=0.5,
                 mz_column_pos=1,
                 rt_column_pos=2,
                 intensity_column_pos=3,
                 min_rt_width=0.000001,
                 min_mz_width=0.000001,
                 n_categories=1):
                 
        """
        TODO: add docstring comment
        Creates a new ROI aligner
        :param mz_tolerance_absolute: ???
        :param mz_tolerance_ppm: ???
        :param rt_tolerance: ???
        :param mz_column_pos: ???
        :param rt_column_pos: ???
        :param intensity_column_pos: ???
        :param min_rt_width: ???
        :param min_mz_width: ???
        :param n_categories: ???
        """

        self.mz_tolerance_absolute, self.mz_tolerance_ppm, self.rt_tolerance = (
            mz_tolerance_absolute, mz_tolerance_ppm, rt_tolerance
        )
        self.mz_column_pos, self.rt_column_pos, self.intensity_column_pos = (
            mz_column_pos, rt_column_pos, intensity_column_pos
        )
        self.min_rt_width, self.min_mz_width = min_rt_width, min_mz_width

        self.n_categories = n_categories
        self.peaksets, self.files_loaded, self.list_of_boxes = [], [], []
        self.sample_names, self.sample_types = [], []
        self.mz_weight, self.rt_weight = 75, 25
        self.peaksets2boxes, self.peaksets2fragintensities = {}, {}
        self.addition_method = None

    def add_sample(self, rois, sample_name, sample_type=None, rt_shifts=None,
                   mz_shifts=None):
        """
        TODO: add docstring comment
        Adds a new sample for alignment
        :param rois: ???
        :param sample_name: ???
        :param sample_type: ???
        :param rt_shifts: ???
        :param mz_shifts: ???
        """
        self.sample_names.append(sample_name)
        self.sample_types.append(sample_type)
        
        these_peaks, frag_intensities, temp_boxes = [], [], []
        for i, roi in enumerate(rois):
            source_id = f"{sample_name}_{i}"
            peak_mz = roi.get_mean_mz()
            peak_rt = roi.estimate_apex()
            peak_intensity = roi.get_max_intensity()
            these_peaks.append(
                Peak(peak_mz, peak_rt, peak_intensity, sample_name, source_id))
            frag_intensities.append(roi.max_fragmentation_intensity)
            rt_shift = 0 if rt_shifts is None else rt_shifts[i]
            mz_shift = 0 if mz_shifts is None else mz_shifts[i]
            temp_boxes.append(
                roi.to_box(self.min_rt_width, self.min_mz_width, rt_shift,
                           mz_shift))

        # do alignment, adding the peaks and boxes, and recalculating max
        # frag intensity
        self._align(these_peaks, temp_boxes, frag_intensities, sample_name)
        
    @staticmethod
    def load_boxes(peak_file, picking_method):
        if picking_method == "mzmine":
            boxes = load_picked_boxes(peak_file)
        elif picking_method == "peakonly":
            boxes = load_peakonly_boxes(peak_file)  # not tested
        elif picking_method == "xcms":
            boxes = load_xcms_boxes(peak_file)  # not tested
        else:
            raise NotImplementedError(f"Picking method \"{picking_method}\" not recognised!")
        return boxes

    def add_picked_peaks(self, mzml_file, peak_file, sample_name,
                         picking_method="mzmine", sample_type=None,
                         half_isolation_window=0, allow_last_overlap=False,
                         rt_shifts=None, mz_shifts=None):
        """
        TODO: add docstring comment
        Adds picked peak information to the aligner
        :param mzml_file: ???
        :param peak_file: ???
        :param sample_name: ???
        :param picking_method: ???
        :param sample_type: ???
        :param half_isolation_window: ???
        :param allow_last_overlap: ???
        :param rt_shifts: ???
        :param mz_shifts: ???
        """
        self.sample_names.append(sample_name)
        self.sample_types.append(sample_type)
        
        these_peaks, frag_intensities = [], []
        temp_boxes = self.load_boxes(peak_file, picking_method)
        temp_boxes = update_picked_boxes(temp_boxes, rt_shifts, mz_shifts)
        self.list_of_boxes.append(temp_boxes)
        
        mzml = path_or_mzml(mzml_file)
        scans2boxes, boxes2scans = map_boxes_to_scans(
                                        mzml, 
                                        temp_boxes, 
                                        half_isolation_window=half_isolation_window,
                                        allow_last_overlap=allow_last_overlap
                                    )
        precursor_intensities, scores = (
            get_precursor_intensities(boxes2scans, temp_boxes, "max")
        )
        
        for i, box in enumerate(temp_boxes):
            source_id = f"{sample_name}_{i}"
            peak_mz = box.mz
            peak_rt = box.rt_in_seconds
            these_peaks.append(
                Peak(peak_mz, peak_rt, box.height, sample_name, source_id))
            frag_intensities.append(precursor_intensities[i])

        self._align(these_peaks, temp_boxes, frag_intensities, sample_name)

    def _align(self, these_peaks, temp_boxes, frag_intensities, short_name):
        """
        TODO: add docstring comment
        Performs alignment of ...
        :param these_peaks: ???
        :param temp_boxes: ???
        :param frag_intensities: ???
        :param short_name: ???
        """
        seen_ps, unassigned = set(), []
        for peak, box, intensity in zip(these_peaks, temp_boxes, frag_intensities):
            candidates = [
                ps for ps in self.peaksets 
                if not ps in seen_ps and ps.is_in_box(peak, 
                                                      self.mz_tolerance_absolute, 
                                                      self.mz_tolerance_ppm, 
                                                      self.rt_tolerance
                                                     )
            ]
            if(len(candidates) > 0):
                scores = [ps.compute_weight(peak, self.mz_tolerance_absolute, self.mz_tolerance_ppm, self.rt_tolerance, self.mz_weight, self.rt_weight) for ps in candidates]
                best_ps, _ = max(((ps, s) for ps, s in zip(candidates, scores)), key=lambda t: t[1])
                best_ps.add_peak(peak)
                self.peaksets2boxes[best_ps].append(box)
                self.peaksets2fragintensities[best_ps].append(intensity)
                seen_ps.add(best_ps)
            else:
                unassigned.append((peak, box, intensity))
        
        for peak, box, intensity in unassigned:
            new_ps = PeakSet(peak)
            self.peaksets.append(new_ps)
            self.peaksets2boxes[new_ps] = [box]
            self.peaksets2fragintensities[new_ps] = [intensity]
            
        self.files_loaded.append(short_name)

    def to_matrix(self):
        """
        Converts aligned peaksets to nicely formatted intensity matrix
        (rows: peaksets, columns: files)
        """
        return np.array(
            [
                [peakset.get_intensity(filename) for fname in self.files_loaded] 
                for ps in self.peaksets
            ], 
            dtype=np.double
        )

    def get_boxes(self, method="mean"):
        """
        Converts peaksets to generic boxes
        """
        if method == "max": f1, f2 = min, max
        else: f1 = f2 = mean
        
        boxes = []
        for ps in self.peaksets:
            box_list = self.peaksets2boxes[ps]
            x1 = f1(b.pt1.x for b in box_list)
            x2 = f2(b.pt2.x for b in box_list)
            y1 = f1(b.pt1.y for b in box_list)
            y2 = f2(b.pt2.y for b in box_list)
            intensity = max(self.peaksets2fragintensities[ps])
            boxes.append(
                GenericBox(x1, x2, y1, y2, 
                           intensity=intensity, 
                           min_xwidth=self.min_rt_width,
                           min_ywidth=self.min_mz_width
                )
            )
        return boxes

    def get_max_frag_intensities(self):
        """
        Returns the maximum fragmentation intensities of peaksets
        """
        return [max(self.peaksets2fragintensities[ps]) for ps in self.peaksets]


class FrequentistRoiAligner(RoiAligner):
    """
    TODO: add docstring comment
    This class does ...
    """

    def get_boxes(self, method='mean'):
        """
        TODO: add docstring comment
        Converts peaksets to generic boxes in a different way
        """
        boxes = super().get_boxes(method)
        categories = np.unique(np.array(self.sample_types))
        enough_categories = min(
            Counter(self.sample_types).values()) > 1 and len(
            categories) == self.n_categories
        pvalues = self.get_p_values(enough_categories)
        for i, box in enumerate(boxes):
            box.pvalue = pvalues[i]
        return boxes

    def get_p_values(self, enough_catergories):
        """
        TODO: add docstring comment
        Returns the p-values of ...
        """
        # need to match boxes, not base chemicals
        if enough_catergories:
            p_values = []
            # sort X
            X = np.array(self.to_matrix())
            # sort y
            categories = np.unique(np.array(self.sample_types))
            if self.n_categories == 2:  # logistic regression
                x = np.array([1 for i in self.sample_types])
                if 'control' in categories:
                    control_type = 'control'
                else:
                    control_type = categories[0]
                x[np.where(np.array(self.sample_types) == control_type)] = 0
                x = sm.add_constant(x)
                for i in range(X.shape[0]):
                    y = np.log(X[i, :] + 1)
                    model = sm.OLS(y, x)
                    p_values.append(model.fit(disp=0).pvalues[1])
            else:  # classification
                pass
        else:
            p_values = [None for ps in self.peaksets]
        return p_values


###############################################################################
# Other useful methods related to ROIs
###############################################################################

# Make the RoI from an input file
# mz_units = Da for Daltons
# mz_units = ppm for ppm
def make_roi(input_file, roi_params):
    """
    Make ROIs from an input file
    :param input_file: input mzML file
    :param roi_params: an RoiParam object
    """

    run = pymzml.run.Reader(input_file, MS1_Precision=5e-6,
                            extraAccessions=[
                                ('MS:1000016', ['value', 'unitName'])],
                            obo_version='4.0.1')

    scan_id = 0
    roi_builder = RoiBuilder(
        roi_params.mz_tol, 0, roi_params.min_intensity,
        roi_params.min_length,
        mz_units=roi_params.mz_units,
        length_units=roi_params.length_units,
        min_roi_intensity_for_fragmentation=roi_params.min_roi_intensity_for_fragmentation)
    for i, spectrum in enumerate(run):
        ms_level = 1
        if spectrum['ms level'] == ms_level:
            current_ms1_scan_rt, units = spectrum.scan_time

            # check that ms1 scan (in seconds) is within bound
            if units == 'minute':
                current_ms1_scan_rt *= 60.0
            if current_ms1_scan_rt < roi_params.start_rt:
                continue
            if current_ms1_scan_rt > roi_params.stop_rt:
                break

            # get the raw peak data from spectrum
            mzs, intensities = spectrum_to_arrays(spectrum)

            # update the ROI construction based on the new scan
            scan = Scan(scan_id, mzs, intensities, ms_level,
                        current_ms1_scan_rt)
            roi_builder.update_roi(scan)
            scan_id += 1

    good_roi = roi_builder.get_good_rois()
    return good_roi


def spectrum_to_arrays(spectrum):
    """
    Converts pymzml spectrum to parallel arrays
    :param spectrum: a pymzml spectrum object
    :return a tuple (mzs, intensities) where mzs and intensities are numpy
    arrays of m/z and intensity values
    """
    mzs = []
    intensities = []
    for mz, intensity in spectrum.peaks('raw'):
        mzs.append(mz)
        intensities.append(intensity)
    mzs = np.array(mzs)
    intensities = np.array(intensities)
    return mzs, intensities


# flake8: noqa: C901
def match(mz, roi_list, mz_tol, mz_units=MZ_UNITS_PPM):
    """
    # Find the RoI that a particular mz falls into. If it falls into nothing,
    return None.
    :param mz: an ROI object containing the m/z we want to find
    :param roi_list: the list of other ROIs to determine where to place the
    mz ROI into
    :param mz_tol: m/z tolerance. This is the window above and below the
    mean_mz of the RoI.
                   E.g. if mz_tol = 1 Da, then it looks plus and minus 1Da
    :param mz_units: units for tolerance, in either 'Da' or 'ppm'
    """
    assert mz_units in [MZ_UNITS_PPM, MZ_UNITS_DA]

    if len(roi_list) == 0:
        return None
    pos = bisect.bisect_right(roi_list, mz)

    if pos == len(roi_list):
        if mz_units == MZ_UNITS_DA:
            dist_left = mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()
        else:  # ppm
            dist_left = 1e6 * (mz.get_mean_mz() - roi_list[
                pos - 1].get_mean_mz()) / mz.get_mean_mz()

        if dist_left < mz_tol:
            return roi_list[pos - 1]
        else:
            return None
    elif pos == 0:
        if mz_units == MZ_UNITS_DA:
            dist_right = roi_list[pos].get_mean_mz() - mz.get_mean_mz()
        else:  # ppm
            dist_right = 1e6 * (roi_list[
                                    pos].get_mean_mz() - mz.get_mean_mz()) / mz.get_mean_mz()

        if dist_right < mz_tol:
            return roi_list[pos]
        else:
            return None
    else:
        if mz_units == MZ_UNITS_DA:
            dist_left = mz.get_mean_mz() - roi_list[pos - 1].get_mean_mz()
            dist_right = roi_list[pos].get_mean_mz() - mz.get_mean_mz()
        else:  # ppm
            dist_left = 1e6 * (mz.get_mean_mz() - roi_list[
                pos - 1].get_mean_mz()) / mz.get_mean_mz()
            dist_right = 1e6 * (roi_list[
                                    pos].get_mean_mz() - mz.get_mean_mz()) / mz.get_mean_mz()

        if dist_left < mz_tol and dist_right > mz_tol:
            return roi_list[pos - 1]
        elif dist_left > mz_tol and dist_right < mz_tol:
            return roi_list[pos]
        elif dist_left < mz_tol and dist_right < mz_tol:
            if dist_left <= dist_right:
                return roi_list[pos - 1]
            else:
                return roi_list[pos]
        else:
            return None


def roi_correlation(roi1, roi2, min_rt_point_overlap=5, method='pearson'):
    """
    Computes the correlation between two ROI objects
    :param roi1: first ROI
    :param roi2: second ROI
    :param min_rt_point_overlap: minimum points that overlap in RT,
    currently unused
    :param correlation method, if 'pearson' then Peason's correlation is
    used, otherwise the cosine score is used
    """
    # flip around so that roi1 starts earlier (or equal)
    if roi2.rt_list[0] < roi1.rt_list[0]:
        temp = roi2
        roi2 = roi1
        roi1 = temp

    # check that they meet the min_rt_point overlap
    if roi1.rt_list[-1] < roi2.rt_list[0]:
        # no overlap at all
        return 0.0

    # find the position of the first element in roi2 in roi1
    pos = roi1.rt_list.index(roi2.rt_list[0])

    # print roi1.rt_list
    # print roi2.rt_list
    # print pos

    total_length = max([len(roi1.rt_list), len(roi2.rt_list) + pos])
    # print total_length

    r1 = np.zeros((total_length), np.double)
    r2 = np.zeros_like(r1)

    r1[:len(roi1.rt_list)] = roi1.intensity_list
    r2[pos:pos + len(roi2.rt_list)] = roi2.intensity_list

    if method == 'pearson':
        r, _ = pearsonr(r1, r2)
    else:
        r = cosine_score(r1, r2)

    return r


def cosine_score(u, v):
    """
    Computes the cosine similarity between two vectors
    :param u: first vector
    :param v: second vector
    """
    numerator = (u * v).sum()
    denominator = np.sqrt((u * u).sum()) * np.sqrt((v * v).sum())
    return numerator / denominator


def greedy_roi_cluster(roi_list, corr_thresh=0.75, corr_type='cosine'):
    """
    Performs a greedy clustering of ROIs
    :param roi_list: a list of ROIs
    :param corr_thresh: the threshold on correlation for clustering
    :param corr_type: correlation type, currently unused
    """
    # sort in descending intensity
    roi_list_copy = [r for r in roi_list]
    roi_list_copy.sort(key=lambda x: max(x.intensity_list), reverse=True)
    roi_clusters = []
    while len(roi_list_copy) > 0:
        roi_clusters.append([roi_list_copy[0]])
        remove_idx = [0]
        if len(roi_list_copy) > 1:
            for i, r in enumerate(roi_list_copy[1:]):
                corr = roi_correlation(roi_list_copy[0], r)
                if corr > corr_thresh:
                    roi_clusters[-1].append(r)
                    remove_idx.append(i + 1)
        remove_idx.sort(reverse=True)
        for r in remove_idx:
            del roi_list_copy[r]

    return roi_clusters


def plot_roi(roi, statuses=None, log=False):
    """
    Plots an ROI
    :param roi: the ROI to plot
    :param statuses: flags for coloring
    :param log: whether to log the intensity (defaults False)
    """
    if log:
        intensities = np.log(roi.intensity_list)
        plt.ylabel('Log Intensity')
    else:
        intensities = roi.intensity_list
        plt.ylabel('Intensity')
    if statuses is not None:
        colours = []
        for s in statuses:
            if s == 'Noise':
                colours.append('red')
            elif s == 'Increase':
                colours.append('blue')
            elif s == 'Decrease':
                colours.append('yellow')
            else:
                colours.append('green')
        plt.scatter(roi.rt_list, intensities, color=colours)
    else:
        plt.scatter(roi.rt_list, intensities)
    plt.xlabel('RT')
    plt.show()


def update_picked_boxes(picked_boxes, rt_shifts, mz_shifts):
    """
    Updates picked boxes ??
    TODO: add docstring comment

    :param picked_boxes: ???
    :param rt_shifts: ???
    :param mz_shifts: ???
    """
    if rt_shifts is None and mz_shifts is None:
        return picked_boxes
        
    new_boxes = copy.deepcopy(picked_boxes)
    for box, sec_shift, mz_shift in zip(new_boxes, rt_shifts, min_shifts):
        if rt_shifts is not None:
            sec_shift = float(sec_shift)
            min_shift = sec_shift / 60.0
        
            box.rt += min_shift
            box.rt_in_minutes += min_shift
            box.rt_in_seconds += sec_shift
            box.rt_range = [r + min_shift for r in box.rt_range]
            box.rt_range_in_seconds = [r + sec_shift for r in box.rt_range_in_seconds]
            
        if mz_shifts is not None:
            mz_shift = float(mz_shift)
            box.mz += mz_shift
            box.mz_range = [r + mz_shift for r in box.mz_range]
