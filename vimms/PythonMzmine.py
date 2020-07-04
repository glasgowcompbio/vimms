import os
import xml.etree.ElementTree

import numpy as np
import pandas as pd
from loguru import logger

from vimms.Chemicals import UnknownChemical
from vimms.Common import PROTON_MASS
from vimms.MassSpec import ScanParameters
from vimms.PlotsForPaper import get_chem_frag_counts, update_matched_status, compute_pref_rec_f1, get_frag_events
from vimms.SpectralUtils import get_scans


def pick_peaks(file_list,
               xml_template='batch_files/PretermPilot2Reduced.xml',
               output_dir='/Users/simon/git/pymzmine/output',
               mzmine_command='/Users/simon/MZmine-2.40.1/startMZmine_MacOSX.command',
               add_name=None):
    et = xml.etree.ElementTree.parse(xml_template)
    # Loop over files in the list (just the firts three for now)
    for filename in file_list:
        logger.info("Creating xml batch file for {}".format(filename.split(os.sep)[-1]))
        root = et.getroot()
        for child in root:
            # Set the input filename
            if child.attrib['method'] == 'net.sf.mzmine.modules.rawdatamethods.rawdataimport.RawDataImportModule':
                for e in child:
                    for g in e:
                        g.text = filename  # raw data file name
            # Set the csv export filename
            if child.attrib[
                'method'] == 'net.sf.mzmine.modules.peaklistmethods.io.csvexport.CSVExportModule':  # TODO: edit / remove
                for e in child:
                    for g in e:
                        tag = g.tag
                        text = g.text
                        if tag == 'current_file' or tag == 'last_file':
                            if add_name is None:
                                csv_name = os.path.join(output_dir,
                                                        filename.split(os.sep)[-1].split('.')[0] + '_pp.csv')
                            else:
                                csv_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[
                                    0] + '_' + add_name + '_pp.csv')
                            g.text = csv_name
        # write the xml file for this input file
        if add_name is None:
            new_xml_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[0] + '.xml')
        else:
            new_xml_name = os.path.join(output_dir, filename.split(os.sep)[-1].split('.')[0] + '_' + add_name + '.xml')
        et.write(new_xml_name)
        # Run mzmine
        logger.info("Running mzMine for {}".format(filename.split(os.sep)[-1]))
        os.system(mzmine_command + ' "{}"'.format(new_xml_name))


def pick_peaks2chems(csv_file):
    df = pd.read_csv(csv_file)
    rts = df['row retention time'] * 60
    mzs = df['row m/z'] - PROTON_MASS
    chems = []
    for i in range(len(rts)):
        chem = UnknownChemical(mzs[i], rts[i], max_intensity=0, chromatogram=None, children=None)
        chems.append(chem)
    return chems


def mzmine_score(controller, dataset, ms1_chems, ms2_chems, min_ms1_intensity, matching_mz_tol, matching_rt_tol):
    chem_to_frag_events = get_frag_events(controller, 2)

    # match with xcms peak-picked ms1 data from fullscan file
    matches_fullscan = mzmine_match(dataset, ms1_chems, matching_mz_tol, matching_rt_tol, verbose=False)

    # match with xcms peak-picked ms1 data from fragmentation file
    matches_fragfile = mzmine_match(dataset, ms2_chems, matching_mz_tol, matching_rt_tol, verbose=False)

    # check if matched and set a flag to indicate that
    update_matched_status(dataset, matches_fullscan, matches_fragfile)

    # True positive: a peak that is fragmented above the minimum MS1 intensity and is picked by XCMS from
    # the MS1 information in the DDA file and is picked in the fullscan file.
    found_in_both = list(filter(lambda x: x.found_in_fullscan and x.found_in_fragfile, dataset))
    frag_count = get_chem_frag_counts(found_in_both, chem_to_frag_events, min_ms1_intensity)
    tp = [chem for chem in found_in_both if frag_count[chem]['good'] > 0 and frag_count[chem]['bad'] == 0]
    tp = len(tp)

    # False positive: any peak that is above minimum intensity and is picked by XCMS
    # from the DDA file but is not picked from the fullscan.
    found_in_dda_only = list(filter(lambda x: not x.found_in_fullscan and x.found_in_fragfile, dataset))
    frag_count = get_chem_frag_counts(found_in_dda_only, chem_to_frag_events, min_ms1_intensity)
    fp = [chem for chem in found_in_dda_only if frag_count[chem]['good'] > 0 and frag_count[chem]['bad'] == 0]
    fp = len(fp)

    # False negative: any peak that is picked from fullscan data, and is not fragmented, or
    # is fragmented below the minimum intensity.
    found_in_fullscan = list(filter(lambda x: x.found_in_fullscan, dataset))
    fn = len(found_in_fullscan) - tp
    if tp == 0:
        prec = rec = f1 = 0
    else:
        prec, rec, f1 = compute_pref_rec_f1(tp, fp, fn)
    return tp, fp, fn, prec, rec, f1


def controller_score(controller, dataset, ms1_picked_peaks_file, ms2_picked_peaks_file, min_ms1_intensity,
                     matching_mz_tol, matching_rt_tol):
    # convert to chemicals
    ms1_chems = pick_peaks2chems(ms1_picked_peaks_file)
    ms2_chems = pick_peaks2chems(ms2_picked_peaks_file)
    # calculate score
    tp, fp, fn, prec, rec, f1 = mzmine_score(controller, dataset, ms1_chems, ms2_chems, min_ms1_intensity,
                                             matching_mz_tol, matching_rt_tol)
    return prec, rec, f1


def mzmine_find_chem(to_find, min_rts, max_rts, min_mzs, max_mzs, chem_list):
    query_mz = to_find.isotopes[0][0]
    query_rt = to_find.chromatogram.rts[to_find.chromatogram.intensities.argmax()] + to_find.rt
    min_rt_check = min_rts <= query_rt
    max_rt_check = query_rt <= max_rts
    min_mz_check = min_mzs <= query_mz
    max_mz_check = query_mz <= max_mzs
    idx = np.nonzero(min_rt_check & max_rt_check & min_mz_check & max_mz_check)[0]
    matches = chem_list[idx]

    # pick a match
    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:  # multiple matches, take the closest in rt
        diffs = [np.abs(chem.rt - to_find.rt) for chem in matches]
        idx = np.argmin(diffs)
        return matches[idx]


def mzmine_match(chemical_list_1, chemical_list_2, mz_tol, rt_tol, verbose=False):
    matches = {}
    missing = []
    chem_list = np.array(chemical_list_2)
    min_rts = np.array([chem.rt - rt_tol for chem in chem_list])
    max_rts = np.array([chem.rt + rt_tol for chem in chem_list])
    min_mzs = np.array([chem.isotopes[0][0] * (1 - mz_tol / 1e6) for chem in chem_list])
    max_mzs = np.array([chem.isotopes[0][0] * (1 + mz_tol / 1e6) for chem in chem_list])
    for i in range(len(chemical_list_1)):
        to_find = chemical_list_1[i]
        if i % 1000 == 0 and verbose:
            logger.debug('%d/%d found %d' % (i, len(chemical_list_1), len(matches)))
        match = mzmine_find_chem(to_find, min_rts, max_rts, min_mzs, max_mzs, chem_list)
        if match:
            matches[to_find] = match
    return matches


def find_col(df, name):
    locations = np.array([colname.find(name) for colname in df.columns])
    return np.array(df[df.columns[np.where(locations != -1)[0]][0]])


def get_base_scoring_df(ms1_picked_peaks_file=None, ms2_picked_peaks_file=None):
    if ms1_picked_peaks_file is not None:
        ms1_df = pd.read_csv(ms1_picked_peaks_file)
        # get MS1 picked peak information
        ms1_scoring_df = pd.DataFrame({'m/z min': find_col(ms1_df, 'Peak m/z min')})
        ms1_scoring_df['m/z max'] = find_col(ms1_df, 'Peak m/z max')
        ms1_scoring_df['rt min'] = find_col(ms1_df, 'Peak RT start') * 60
        ms1_scoring_df['rt max'] = find_col(ms1_df, 'Peak RT end') * 60
        ms1_scoring_df['rt centre'] = find_col(ms1_df, 'row retention time') * 60
    if ms2_picked_peaks_file is not None:
        ms2_df = pd.read_csv(ms2_picked_peaks_file)
        # get MS2 picked peak information
        ms2_scoring_df = pd.DataFrame({'m/z min': find_col(ms2_df, 'Peak m/z min')})
        ms2_scoring_df['m/z max'] = find_col(ms2_df, 'Peak m/z max')
        ms2_scoring_df['rt min'] = find_col(ms2_df, 'Peak RT start') * 60
        ms2_scoring_df['rt max'] = find_col(ms2_df, 'Peak RT end') * 60
        ms2_scoring_df['rt centre'] = find_col(ms2_df, 'row retention time') * 60
    # return
    if ms1_picked_peaks_file is None and ms2_picked_peaks_file is None:
        return None
    elif ms1_picked_peaks_file is None:
        return ms2_scoring_df
    elif ms2_picked_peaks_file is None:
        return ms1_scoring_df
    else:
        return ms1_scoring_df, ms2_scoring_df


def get_max_intensity(controller, dataset, rt, ms1_isolation_window):
    mz_int_pairs = []
    for chem in dataset:
        new_pair = controller.environment.mass_spec._get_all_mz_peaks(chem, rt, 1, ms1_isolation_window)
        if new_pair is not None:
            mz_int_pairs.append(new_pair[0])
    if len(mz_int_pairs) == 0:
        return 0
    max_int = max([i[1] for i in mz_int_pairs])
    return max_int


def peak_scoring(controller, ms1_picked_peaks_file, ms2_picked_peaks_file, dataset, min_ms1_intensity,
                 score_param_dict):
    ms1_scoring_df, ms2_scoring_df = get_base_scoring_df(ms1_picked_peaks_file, ms2_picked_peaks_file)
    scoring_count = [0 for i in range(len(ms1_scoring_df.index))]
    scoring_logint = [0 for i in range(len(ms1_scoring_df.index))]

    # get combined (full scans + frag scans), sorted by rt ascending
    full_scans = controller.scans[1]
    frag_scans = controller.scans[2]
    scans = full_scans + frag_scans
    scans = sorted(scans, key=lambda scan: scan.rt)

    last_ms1_scan = None
    for scan in scans:
        if scan.ms_level == 1:
            last_ms1_scan = scan

        if scan.ms_level == 2:
            query_mz_window = scan.scan_params.compute_isolation_windows()[0][0]
            query_mz = scan.scan_params.get(ScanParameters.PRECURSOR_MZ).precursor_mz
            query_rt = last_ms1_scan.rt  # query RT is the RT of the last ms1 scan
            max_intensity = get_max_intensity(controller, dataset, query_rt,
                                              scan.scan_params.compute_isolation_windows())

            if max_intensity > min_ms1_intensity:
                # check whether in an MS1 peak - record peaks it is in
                # not sure which one is better between check_boxes_intersection or check_single_box
                # idx_ms1 = check_boxes_intersection(ms1_scoring_df, query_mz_window, query_rt)
                idx_ms1 = check_single_box(ms1_scoring_df, query_mz, query_rt)

                # check whether in an MS1 peak - record peaks it is in
                # not sure which one is better between check_boxes_intersection or check_single_box
                # idx_ms2 = check_boxes_intersection(ms2_scoring_df, query_mz_window, query_rt)
                idx_ms2 = check_single_box(ms2_scoring_df, query_mz, query_rt)

                # record scores
                if len(idx_ms2) > 0 and len(idx_ms1) > 0:
                    for i in range(len(idx_ms1)):
                        scoring_count[idx_ms1[i]] = 1
                        scoring_logint[idx_ms1[i]] = max(scoring_logint[idx_ms1[i]], np.log(max_intensity))

    score_count = sum(scoring_count) / len(scoring_count)
    score_logint = sum(scoring_logint)
    prec, rec, f1 = controller_score(controller, dataset, ms1_picked_peaks_file, ms2_picked_peaks_file,
                                     min_ms1_intensity, score_param_dict['matching_mz_tol'],
                                     score_param_dict['matching_rt_tol'])
    return score_count, score_logint, prec, rec, f1


def multi_peak_scoring_mzml(ms2_mzml_file, ms1_picked_peaks_file=None, ms2_picked_peaks_file=None,
                            min_ms1_intensity=1.75E5, score_param_dict=None,
                            evaluation_methods=['mzmine_peak', 'mzmine_logint', 'mzmine_f1']):
    if evaluation_methods != ['mzmine_peak']:
        print('Only mzmine_peak currently implemented. Evulating this method regardless')
        evaluation_methods = ['mzmine_peak']

    # check there is a picked peaks file
    if ms1_picked_peaks_file is None and ms2_picked_peaks_file is None:
        print('No picked peaks file provided')
        return None

    f1 = 0

    # set up scoring
    if ms1_picked_peaks_file is not None:
        if ms2_picked_peaks_file is not None:
            ms1_scoring_df, ms2_scoring_df = get_base_scoring_df(ms1_picked_peaks_file, ms2_picked_peaks_file)
        else:
            ms1_scoring_df = get_base_scoring_df(ms1_picked_peaks_file, ms2_picked_peaks_file)
        scoring_count = [0 for i in range(len(ms1_scoring_df.index))]
        scoring_logint = [0 for i in range(len(ms1_scoring_df.index))]
    else:
        ms2_scoring_df = get_base_scoring_df(ms1_picked_peaks_file, ms2_picked_peaks_file)
        scoring_count = [0 for i in range(len(ms2_scoring_df.index))]
        scoring_logint = [0 for i in range(len(ms2_scoring_df.index))]

    scans = get_scans(ms2_mzml_file)
    last_ms1_scan = None
    for scan in scans:
        if scan.ms_level == 1:
            last_ms1_scan = scan

        if scan.ms_level == 2:
            # get isolation window
            query_mz_window = scan.scan_params.compute_isolation_windows()[0][0]  # TODO: check this is right
            query_mz = scan.scan_params.get(ScanParameters.PRECURSOR_MZ).precursor_mz
            query_rt = last_ms1_scan.rt  # query_rt should be the RT of the last ms1 scan, not the current ms2 scan!

            # check above max intensity
            # TODO: work out max intensity of any peak fragmented in that isolation window
            # max_intensity = get_max_intensity(controller, dataset, scan.rt, scan.scan_params.compute_isolation_windows())
            # if max_intensity > min_ms1_intensity:

            if ms1_picked_peaks_file is not None:
                # check whether in an MS1 peak - record peaks it is in
                # not sure which one is better between check_boxes_intersection or check_single_box
                # idx_ms1 = check_boxes_intersection(ms1_scoring_df, query_mz_window, query_rt)
                idx_ms1 = check_single_box(ms1_scoring_df, query_mz, query_rt)

            if ms2_picked_peaks_file is not None:
                # check whether in an MS1 peak - record peaks it is in
                # not sure which one is better between check_boxes_intersection or check_single_box
                # idx_ms2 = check_boxes_intersection(ms2_scoring_df, query_mz_window, query_rt)
                idx_ms2 = check_single_box(ms2_scoring_df, query_mz, query_rt)

            if ms1_picked_peaks_file is not None and ms2_picked_peaks_file is not None:
                if len(idx_ms2) > 0 and len(idx_ms1) > 0:
                    for i in range(len(idx_ms1)):
                        scoring_count[idx_ms1[i]] = 1
                        # TODO: add logint bit
                        # scoring_logint[idx_ms1[i]] = max(scoring_logint[idx_ms1[i]], np.log(max_intensity))

            if ms1_picked_peaks_file is not None:
                if len(idx_ms1) > 0:
                    for i in range(len(idx_ms1)):
                        scoring_count[idx_ms1[i]] = 1
                        # TODO: add logint bit
                        # scoring_logint[idx_ms1[i]] = max(scoring_logint[idx_ms1[i]], np.log(max_intensity))

            if ms2_picked_peaks_file is not None:
                if len(idx_ms2) > 0:
                    for i in range(len(idx_ms2)):
                        scoring_count[idx_ms2[i]] = 1
                        # TODO: add logint bit
                        # scoring_logint[idx_ms2[i]] = max(scoring_logint[idx_ms2[i]], np.log(max_intensity))

    # add f1 if required
    if 'mzmine_f1' in evaluation_methods:
        f1 = 0  # TODO: needs implementing
        # prec, rec, f1 = controller_score(controller, dataset, ms1_picked_peaks_file, ms2_picked_peaks_file,
        #                                  min_ms1_intensity, score_param_dict['matching_mz_tol'],
        #                                  score_param_dict['matching_rt_tol'])

    results_dict = {'mzmine_peak': (sum(scoring_count) / len(scoring_count)),
                    'mzmine_logint': sum(scoring_logint),
                    'mzmine_f1': f1}
    return results_dict


def check_boxes_intersection(scoring_df, query_mz_window, query_rt):
    rt_check = (scoring_df['rt min'] <= query_rt) & (query_rt <= scoring_df['rt max'])
    mz_check1 = (scoring_df['m/z min'] >= query_mz_window[0]) & (query_mz_window[1] >= scoring_df['m/z min'])
    mz_check2 = (scoring_df['m/z max'] >= query_mz_window[0]) & (query_mz_window[1] >= scoring_df['m/z max'])
    mz_check3 = (scoring_df['m/z min'] <= query_mz_window[0]) & (scoring_df['m/z max'] >= query_mz_window[1])
    mz_check = mz_check1 | mz_check2 | mz_check3
    indices = np.nonzero(np.array(rt_check) & np.array(mz_check))[0]
    return indices


def check_single_box(scoring_df, query_mz, query_rt):
    rt_check = (scoring_df['rt min'] <= query_rt) & (query_rt <= scoring_df['rt max'])
    mz_check = (scoring_df['m/z min'] <= query_mz) & (scoring_df['m/z max'] >= query_mz)
    indices = np.nonzero(np.array(rt_check) & np.array(mz_check))[0]
    return indices
