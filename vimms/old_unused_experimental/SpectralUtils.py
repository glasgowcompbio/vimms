# Collection of methods to deal with mass spectra mzML files
import numpy as np
import pandas as pd
import pymzml
from loguru import logger

from vimms.old_unused_experimental.Chemicals import RoiToChemicalCreator
from vimms.Common import get_rt, ScanParameters, Precursor
from vimms.MassSpec import Scan
from vimms.Roi import make_roi


########################################################################################################################
# Data extraction methods
########################################################################################################################


def get_chemicals(mzML_file, mz_tol, min_ms1_intensity, start_rt, stop_rt, min_length=1):
    '''
    Extract ROI from an mzML file and turn them into UnknownChemical objects
    :param mzML_file: input mzML file
    :param mz_tol: mz tolerance for ROI extraction
    :param min_ms1_intensity: ROI will only be kept if it has one point above this threshold
    :param start_rt: start RT to extract ROI
    :param stop_rt: end RT to extract ROI
    :return: a list of UnknownChemical objects
    '''
    min_intensity = 0
    good_roi, junk = make_roi(mzML_file, mz_tol=mz_tol, mz_units='ppm', min_length=min_length,
                              min_intensity=min_intensity, start_rt=start_rt, stop_rt=stop_rt)

    # keep ROI that have at least one point above the minimum to fragment threshold
    keep = []
    for roi in good_roi:
        if np.count_nonzero(np.array(roi.intensity_list) > min_ms1_intensity) > 0:
            keep.append(roi)

    ps = None  # old_unused_experimental
    rtcc = RoiToChemicalCreator(ps, keep)
    chemicals = np.array(rtcc.chemicals)
    return chemicals


def get_scans(mzml_file, ms_level=None):
    """
    Get (MS1) precursor peaks and their associated MS2 scans from an mzML file
    :param mzml_file: path to an mzML file
    :return: a pandas dataframe that contains all the ms1 and ms2 information
    """
    run = pymzml.run.Reader(mzml_file, obo_version='4.0.1',
                            MS1_Precision=5e-6,
                            extraAccessions=[('MS:1000016', ['value', 'unitName'])])

    scans = []
    for scan_no, scan in enumerate(run):
        peaklist = _get_peaks(scan)
        mzs = peaklist[:, 0]
        rt = peaklist[0, 1]
        intensities = peaklist[:, 2]

        if scan.ms_level == 1:
            vimms_scan = Scan(scan_no, mzs, intensities, scan.ms_level, rt)
            scans.append(vimms_scan)

        if scan.ms_level == 2:
            precursors = scan.selected_precursors
            if len(precursors) > 0:
                assert len(precursors) == 1  # assume exactly 1 precursor peak for each ms2 scan
                precursor = precursors[0]
                try:
                    precursor_mz = precursor['mz']
                    precursor_intensity = precursor['i']
                    vimms_precursor = Precursor(precursor_mz, precursor_intensity, 1, None)

                    params = ScanParameters()
                    params.set(ScanParameters.PRECURSOR_MZ, vimms_precursor)
                    params.set(ScanParameters.ISOLATION_WIDTH, 1)
                    vimms_scan = Scan(scan_no, mzs, intensities, scan.ms_level, rt, scan_params=params)
                    scans.append(vimms_scan)
                except ValueError as e:
                    logger.warning(e)
                except KeyError as e:
                    continue  # sometimes we can't find the intensity value precursor['i'] in precursors

    if ms_level is not None:  # filter by ms level
        scans = [scan for scan in scans if scan.ms_level == ms_level]
    return scans


def get_precursor_info(fragfile):
    """
    Get (MS1) precursor peaks and their associated MS2 scans from an mzML file
    :param fragfile: path to an mzML file
    :return: a pandas dataframe that contains all the ms1 and ms2 information
    """
    run = pymzml.run.Reader(fragfile, obo_version='4.0.1',
                            MS1_Precision=5e-6,
                            extraAccessions=[('MS:1000016', ['value', 'unitName'])])

    last_ms1_peaklist = None
    last_ms1_scan_no = 0
    isolation_width = 1.0  # Dalton
    data = []
    for scan_no, scan in enumerate(run):
        if scan.ms_level == 1:  # save the last ms1 scan that we've seen
            last_ms1_peaklist = _get_peaks(scan)
            last_ms1_scan_no = scan_no

        # TODO: it's better to use the "isolation window target m/z" field in the mzML file for matching
        precursors = scan.selected_precursors
        if len(precursors) > 0:
            assert len(precursors) == 1  # assume exactly 1 precursor peak for each ms2 scan
            precursor = precursors[0]

            try:
                scan_rt = get_rt(scan)
                precursor_mz = precursor['mz']
                precursor_intensity = precursor['i']
                res = _find_precursor_peaks(precursor, last_ms1_peaklist, last_ms1_scan_no,
                                            isolation_width=isolation_width)
                ms2_peaklist = _get_peaks(scan)
                row = [scan_no, scan_rt, precursor_mz, precursor_intensity, ms2_peaklist]
                row.extend(res)
                data.append(row)
            except ValueError as e:
                logger.warning(e)
            except KeyError as e:
                continue  # sometimes we can't find the intensity value precursor['i'] in precursors

    columns = ['ms2_scan_id', 'ms2_scan_rt', 'ms2_precursor_mz', 'ms2_precursor_intensity', 'ms2_peaklist',
               'ms1_scan_id', 'ms1_scan_rt', 'ms1_mz', 'ms1_intensity']
    df = pd.DataFrame(data, columns=columns)

    # select only rows where we are sure of the matching, i.e. the intensity values aren't too different
    df['intensity_diff'] = np.abs(df['ms2_precursor_intensity'] - df['ms1_intensity'])
    idx = (df['intensity_diff'] < 0.1)
    ms1_df = df[idx]
    return ms1_df


########################################################################################################################
# Private methods
########################################################################################################################

def _get_peaks(spectrum):
    mzs = spectrum.mz
    rts = [get_rt(spectrum)] * len(mzs)
    intensities = spectrum.i
    peaklist = np.stack([mzs, rts, intensities], axis=1)
    return peaklist


def _find_precursor_peaks(precursor, last_ms1_peaklist, last_ms1_scan_no, isolation_width=1.0):
    selected_ms1, selected_ms1_idx = _find_precursor_ms1(precursor, last_ms1_peaklist,
                                                         last_ms1_scan_no, isolation_width)
    selected_ms1_mz = selected_ms1[0]
    selected_ms1_rt = selected_ms1[1]
    selected_ms1_intensity = selected_ms1[2]
    res = [last_ms1_scan_no, selected_ms1_rt, selected_ms1_mz, selected_ms1_intensity]
    return res


def _find_precursor_ms1(precursor, last_ms1_peaklist, last_ms1_scan_no, isolation_width):
    precursor_mz = precursor['mz']
    precursor_intensity = precursor['i']

    # find mz in the last ms1 scan that fall within isolation window
    mzs = last_ms1_peaklist[:, 0]
    diffs = abs(mzs - precursor_mz) < (isolation_width / 2)
    idx = np.nonzero(diffs)[0]

    if len(idx) == 0:  # should never happen!?
        raise ValueError('Cannot find precursor peak (%f, %f) in the last ms1 scan %d' %
                         (precursor_mz, precursor_intensity, last_ms1_scan_no))

    elif len(idx) == 1:  # only one is found
        selected_ms1_idx = idx[0]

    else:  # found multilple possible ms1 peak, select the largest intensity
        possible_ms1 = last_ms1_peaklist[idx, :]
        possible_intensities = possible_ms1[:, 2]
        closest = np.argmax(possible_intensities)
        selected_ms1_idx = idx[closest]

    selected_ms1 = last_ms1_peaklist[selected_ms1_idx, :]
    return selected_ms1, selected_ms1_idx
