from loguru import logger
from mass_spec_utils.data_import.mzmine import load_picked_boxes, map_boxes_to_scans
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Common import MZ_UNITS_PPM
from vimms.Roi import make_roi, RoiParams


def picked_peaks_evaluation(mzml_file, picked_peaks_file):
    boxes = load_picked_boxes(picked_peaks_file)
    mz_file = MZMLFile(mzml_file)
    scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes, half_isolation_window=0)
    return float(len(boxes2scans))


def roi_scoring(mzml_file, mz_tol=10, mz_units=MZ_UNITS_PPM, min_length=3, min_intensity=500):
    mz_file = MZMLFile(mzml_file)
    roi_params = RoiParams(mz_tol=mz_tol, min_length=min_length,
                                  min_intensity=min_intensity, mz_units=mz_units)
    good_roi = make_roi(mzml_file, roi_params)
    roi_roi2scan, roi_scan2roi = match_scans_to_rois(mz_file, good_roi)
    with_scan, without_scan, num_scan = prop_roi_with_scans(roi_roi2scan)
    return dict({'with_scan': with_scan, 'without_scan': without_scan, 'num_scan': num_scan})


def summarise(mz_file_object):
    n_scans = len(mz_file_object.scans)
    n_ms1_scans = len(list(filter(lambda x: x.ms_level == 1, mz_file_object.scans)))
    n_ms2_scans = len(list(filter(lambda x: x.ms_level == 2, mz_file_object.scans)))
    logger.debug("Total scans = {}, MS1 = {}, MS2 = {}".format(n_scans, n_ms1_scans, n_ms2_scans))


def match_scans_to_rois(mz_file_object, roi_list):
    roi2scan = {roi: [] for roi in roi_list}
    scan2roi = {scan: [] for scan in filter(lambda x: x.ms_level == 2, mz_file_object.scans)}
    for scan in mz_file_object.scans:
        if scan.ms_level == 2:
            pmz = scan.precursor_mz
            scan_rt_in_seconds = 60 * scan.previous_ms1.rt_in_minutes
            in_mz_range = list(filter(lambda x: min(x.mz_list) <= pmz <= max(x.mz_list), roi_list))
            in_rt_range = list(filter(lambda x: x.rt_list[0] <= scan_rt_in_seconds <= x.rt_list[-1], in_mz_range))
            for roi in in_rt_range:
                roi2scan[roi].append(scan)
                scan2roi[scan].append(roi)
    return roi2scan, scan2roi


def prop_roi_with_scans(roi2scan):
    with_scan = 0
    without_scan = 0
    for r, scans in roi2scan.items():
        if len(scans) == 0:
            without_scan += 1
        else:
            with_scan += 1
    return with_scan, without_scan, len(roi2scan)
