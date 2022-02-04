import os
from collections import defaultdict

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pylab as plt
import pymzml
import seaborn as sns
from loguru import logger

from vimms.Chemicals import UnknownChemical
from vimms.old_unused_experimental.Chemicals import RoiToChemicalCreator
from vimms.Common import load_obj, PROTON_MASS, find_nearest_index_in_array
from vimms.MassSpec import ScanEvent
from vimms.Roi import make_roi, RoiParams
from vimms.old_unused_experimental.SpectralUtils import get_precursor_info, get_chemicals


def get_N(row):
    if 'T10' in row['filename']:
        return 10
    else:
        return row['filename'].split('_')[3]


def get_dew(row):
    if 'T10' in row['filename']:
        return 15
    else:
        tok = row['filename'].split('_')[5]  # get the dew value in the filename
        return tok.split('.')[0]  # get the part before '.mzML'


def experiment_group(row):
    if 'experiment' in row:
        col_to_check = 'experiment'
    else:
        col_to_check = 'filename'

    if 'beer' in row[col_to_check]:
        return 'beer'
    else:
        return 'urine'


def add_group_column(df):
    df['group'] = df.apply(lambda row: experiment_group(row), axis=1)


def get_df(csv_file, min_ms1_intensity, rt_range, mz_range):
    df = pd.read_csv(csv_file)
    return filter_df(df, min_ms1_intensity, rt_range, mz_range)


def filter_df(df, min_ms1_intensity, rt_range, mz_range):
    # filter by rt range
    if rt_range is not None:
        df = df[(df['rt'] > rt_range[0][0]) & (df['rt'] < rt_range[0][1])]

    # filter by mz range
    if mz_range is not None:
        df = df[(df['rt'] > mz_range[0][0]) & (df['rt'] < mz_range[0][1])]

    # filter by min intensity
    intensity_col = 'maxo'
    if min_ms1_intensity is not None:
        df = df[(df[intensity_col] > min_ms1_intensity)]

    # add log intensity column
    df['log_intensity'] = df.apply(lambda row: np.log(row[intensity_col]), axis=1)

    # add N column
    try:
        df['N'] = df.apply(lambda row: get_N(row), axis=1)
        df[['N']] = df[['N']].astype('int')
    except IndexError:
        pass
    except ValueError:
        df['N'] = df.apply(lambda row: np.nan, axis=1)

    # add group column
    df['group'] = df.apply(lambda row: experiment_group(row), axis=1)
    return df


def make_boxplot(df, x, y, xticklabels, title, outfile=None):
    g = sns.catplot(x=x, y=y, kind='box', data=df)
    g.fig.set_size_inches(10, 3)
    if xticklabels is not None:
        g.set_xticklabels(xticklabels, rotation=90)
    else:
        g.set_xticklabels(rotation=90)
    plt.title(title)
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    plt.show()


def make_hist(df, col_name, file_name, title):
    gb = df.groupby('filename')
    group_df = gb.get_group(file_name)
    vals = group_df[col_name].values
    logger.debug(vals, len(vals))
    _ = plt.hist(vals, bins=100)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def to_chemical(row):
    mz = row['mz'] - PROTON_MASS
    rt = row['rt']
    max_intensity = row['maxo']
    chrom = None
    chem = UnknownChemical(mz, rt, max_intensity, chrom, children=None)
    return chem


def df_to_chemicals(df, filename=None):
    if filename is not None:
        filtered_df = df.loc[df['filename'] == filename]
    else:
        filtered_df = df
    chems = filtered_df.apply(lambda row: to_chemical(row), axis=1).values
    return chems


def find_chem(to_find, min_rts, max_rts, min_mzs, max_mzs, chem_list):
    query_mz = to_find.isotopes[0][0]
    query_rt = to_find.rt
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


def match(chemical_list_1, chemical_list_2, mz_tol, rt_tol, verbose=False):
    matches = {}
    chem_list = np.array(chemical_list_2)
    min_rts = np.array([chem.rt - rt_tol for chem in chem_list])
    max_rts = np.array([chem.rt + rt_tol for chem in chem_list])
    min_mzs = np.array([chem.isotopes[0][0] * (1 - mz_tol / 1e6) for chem in chem_list])
    max_mzs = np.array([chem.isotopes[0][0] * (1 + mz_tol / 1e6) for chem in chem_list])
    for i in range(len(chemical_list_1)):
        to_find = chemical_list_1[i]
        if i % 1000 == 0 and verbose:
            logger.debug('%d/%d found %d' % (i, len(chemical_list_1), len(matches)))
        match = find_chem(to_find, min_rts, max_rts, min_mzs, max_mzs, chem_list)
        if match:
            matches[to_find] = match
    return matches


def match_peaklist(mz_list_1, rt_list_1, intensity_list_1, mz_list_2, rt_list_2, intensity_list_2, mz_tol, rt_tol):
    if mz_tol is not None:  # create mz range for matching in ppm
        min_mzs = np.array([mz * (1 - mz_tol / 1e6) for mz in mz_list_2])
        max_mzs = np.array([mz * (1 + mz_tol / 1e6) for mz in mz_list_2])

    else:  # create mz ranges by rounding to 2dp
        min_mzs = np.around(mz_list_2, decimals=2)
        max_mzs = np.around(mz_list_2, decimals=2)
        mz_list_1 = np.around(mz_list_1, decimals=2)

    # create rt ranges for matching
    min_rts = np.array([rt - rt_tol for rt in rt_list_2])
    max_rts = np.array([rt + rt_tol for rt in rt_list_2])

    matches = {}
    for i in range(len(mz_list_1)):  # loop over query and find a match
        query = (mz_list_1[i], rt_list_1[i], intensity_list_1[i],)
        match = find_match(query, min_rts, max_rts, min_mzs, max_mzs, mz_list_2, rt_list_2, intensity_list_2)
        matches[query] = match
    return matches


def check_found_matches(matches, left_label, right_label, N=20):
    found = [key for key in matches if matches[key] is not None]
    logger.debug('Found %d/%d (%f)' % (len(found), len(matches), len(found) / len(matches)))

    logger.debug('%s\t\t\t\t\t\t%s' % (left_label, right_label))
    for key, value in list(matches.items())[0:N]:
        if value is not None:
            logger.debug('mz %.2f rt %.4f intensity %.4f\tmz %.2f rt %.4f intensity %.4f' % (
                key[0], key[1], key[2], value[0], value[1], value[2]))


def plot_matched_precursors(matches, min_mz, max_mz, min_rt, max_rt, out_file=None):
    plt.figure(figsize=(12, 6))
    plt.rcParams.update({'font.size': 24})
    for key in matches:
        mz, rt, intensity = key
        if min_mz < mz < max_mz and min_rt < rt < max_rt:
            if matches[key] is not None:
                plt.plot([rt], [mz], marker='.', markersize=5, color='blue', alpha=0.1)
            else:
                plt.plot([rt], [mz], marker='.', markersize=5, color='red', alpha=0.1)

    blue_patch = mpatches.Patch(color='blue', label='Matched')
    red_patch = mpatches.Patch(color='red', label='Unmatched')
    plt.legend(handles=[blue_patch, red_patch])
    plt.title('Matched fragmentation events', fontsize=30)
    plt.xlabel('Retention Time (s)')
    plt.ylabel('m/z')
    plt.tight_layout()
    if out_file is not None:
        plt.savefig(out_file, dpi=300)


def count_stuff(input_file, min_rt, max_rt):
    run = pymzml.run.Reader(input_file, MS1_Precision=5e-6,
                            extraAccessions=[('MS:1000016', ['value', 'unitName'])],
                            obo_version='4.0.1')
    mzs = []
    rts = []
    intensities = []
    count_ms1_scans = 0
    count_ms2_scans = 0
    cumsum_ms1_scans = []
    cumsum_ms2_scans = []
    count_selected_precursors = 0
    for spectrum in run:
        ms_level = spectrum['ms level']
        current_scan_rt, units = spectrum.scan_time
        if units == 'minute':
            current_scan_rt *= 60.0
        if min_rt < current_scan_rt < max_rt:
            if ms_level == 1:
                count_ms1_scans += 1
                cumsum_ms1_scans.append((current_scan_rt, count_ms1_scans,))
            elif ms_level == 2:
                try:
                    selected_precursors = spectrum.selected_precursors
                    count_selected_precursors += len(selected_precursors)
                    mz = selected_precursors[0]['mz']
                    intensity = selected_precursors[0]['i']

                    count_ms2_scans += 1
                    mzs.append(mz)
                    rts.append(current_scan_rt)
                    intensities.append(intensity)
                    cumsum_ms2_scans.append((current_scan_rt, count_ms2_scans,))
                except KeyError:
                    # logger.debug(selected_precursors)
                    pass

    logger.debug('Number of ms1 scans =', count_ms1_scans)
    logger.debug('Number of ms2 scans =', count_ms2_scans)
    logger.debug('Total scans =', count_ms1_scans + count_ms2_scans)
    logger.debug('Number of selected precursors =', count_selected_precursors)
    return np.array(mzs), np.array(rts), np.array(intensities), np.array(cumsum_ms1_scans), np.array(cumsum_ms2_scans)


def find_match(query, min_rts, max_rts, min_mzs, max_mzs, mz_list, rt_list, intensity_list):
    # check ranges
    query_mz, query_rt, query_intensity = query
    min_rt_check = min_rts <= query_rt
    max_rt_check = query_rt <= max_rts
    min_mz_check = min_mzs <= query_mz
    max_mz_check = query_mz <= max_mzs
    idx = np.nonzero(min_rt_check & max_rt_check & min_mz_check & max_mz_check)[0]

    # get mz, rt and intensity of matching indices
    matches_mz = mz_list[idx]
    matches_rt = rt_list[idx]
    matches_intensity = intensity_list[idx]

    if len(idx) == 0:  # no match
        return None

    elif len(idx) == 1:  # single match
        return (matches_mz[0], matches_rt[0], matches_intensity[0],)

    else:  # multiple matches, take the closest in rt
        diffs = [np.abs(rt - query_rt) for rt in matches_rt]
        idx = np.argmin(diffs)
        return (matches_mz[idx], matches_rt[idx], matches_intensity[idx],)


def plot_num_scans(real_cumsum_ms1, real_cumsum_ms2, simulated_cumsum_ms1, simulated_cumsum_ms2, out_file=None):
    plt.plot(real_cumsum_ms1[:, 0], real_cumsum_ms1[:, 1], 'r')
    plt.plot(real_cumsum_ms2[:, 0], real_cumsum_ms2[:, 1], 'b')
    plt.plot(simulated_cumsum_ms1[:, 0], simulated_cumsum_ms1[:, 1], 'r--')
    plt.plot(simulated_cumsum_ms2[:, 0], simulated_cumsum_ms2[:, 1], 'b--')

    plt.legend(['Actual MS1', 'Actual MS2', 'Simulated MS1', 'Simulated MS2'])
    plt.xlabel('Retention Time (s)')
    plt.ylabel('Cumulative sum')
    plt.title('Cumulative number of MS1 and MS2 scans', fontsize=18)
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=300)


def plot_matched_intensities(matched_intensities, unmatched_intensities, out_file=None):
    plt.figure()
    temp1 = plt.hist(np.log(matched_intensities), bins=np.linspace(10, 20, 50), color='blue')
    temp2 = plt.hist(np.log(unmatched_intensities), bins=np.linspace(10, 20, 50), color='red')
    plt.title('Matched precursor intensities')

    blue_patch = mpatches.Patch(color='blue', label='Matched')
    red_patch = mpatches.Patch(color='red', label='Unmatched')
    plt.legend(handles=[blue_patch, red_patch])
    plt.xlabel('log(intensity)')
    plt.ylabel('Precursor count')
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=300)


def load_controller(results_dir, experiment_name, N, rt_tol):
    analysis_name = 'experiment_%s_N_%d_rttol_%d' % (experiment_name, N, rt_tol)
    pickle_in = '%s/%s.p' % (results_dir, analysis_name)
    logger.info('Loading %s' % analysis_name)
    try:
        controller = load_obj(pickle_in)
    except FileNotFoundError:
        controller = None
    return controller


def load_controllers(results_dir, Ns, rt_tols):
    controllers = []
    for N in Ns:
        for rt_tol in rt_tols:
            controller = load_controller(results_dir, N, rt_tol)
            if controller is not None:
                controllers.append(controller)
    return controllers


def compute_performance_scenario_1(controller, dataset, min_ms1_intensity,
                                   fullscan_filename, P_peaks_df,
                                   matching_mz_tol, matching_rt_tol,
                                   chem_to_frag_events=None):
    if chem_to_frag_events is None:  # read MS2 fragmentation events from pickled controller
        chem_to_frag_events = get_frag_events(controller, 2)

    # match with xcms peak-picked ms1 data
    detected_ms1 = df_to_chemicals(P_peaks_df, fullscan_filename)
    matches_fullscan = match(dataset, detected_ms1, matching_mz_tol, matching_rt_tol, verbose=False)

    # check if matched and set a flag to indicate that
    update_matched_status(dataset, matches_fullscan, None)

    # positive instances are ground truth MS1 peaks found by XCMS
    # negative instances are chemicals that cannot be matched to XCMS output
    positives = list(filter(lambda x: x.found_in_fullscan, dataset))
    negatives = list(filter(lambda x: not x.found_in_fullscan, dataset))

    # for both positive and negative instances, count how many frag events they have
    # and whether it's above (good) or below (bad) the minimum ms1 intensity at the time of fragmentation.
    positives_count = get_chem_frag_counts(positives, chem_to_frag_events, min_ms1_intensity)
    negatives_count = get_chem_frag_counts(negatives, chem_to_frag_events, min_ms1_intensity)

    # TP = positive instances that are good only
    tp = [chem for chem in positives if positives_count[chem]['good'] > 0 and positives_count[chem]['bad'] == 0]

    # FP = negative instances that are fragmented (both good + bad)
    fp = [chem for chem in negatives if negatives_count[chem]['good'] > 0 or negatives_count[chem]['bad'] > 0]

    # FN = positive instances that are not fragmented at all + positive instances that are bad only
    fn = [chem for chem in positives if \
          (positives_count[chem]['good'] == 0 and positives_count[chem]['bad'] == 0) or \
          (positives_count[chem]['good'] == 0 and positives_count[chem]['bad'] > 0)]

    tp = len(tp)
    fp = len(fp)
    fn = len(fn)
    prec, rec, f1 = compute_pref_rec_f1(tp, fp, fn)
    return tp, fp, fn, prec, rec, f1


# def compute_performance_scenario_1(controller, chemicals, min_ms1_intensity,
#                                    fullscan_filename, P_peaks_df,
#                                    matching_mz_tol, matching_rt_tol,
#                                    chem_to_frag_events=None):

#     if chem_to_frag_events is None: # read MS2 fragmentation events from pickled controller
#         chem_to_frag_events = get_frag_events(controller, 2)

#     # match xcms picked ms1 peaks to fragmentation peaks
#     detected_ms1 = df_to_chemicals(P_peaks_df, fullscan_filename)
#     matches_fullscan = match(detected_ms1, chemicals, matching_mz_tol, matching_rt_tol, verbose=False)
#     matched_frags = set(matches_fullscan.values())
#     logger.debug('%d/%d %d/%d' % (len(matches_fullscan), len(detected_ms1), len(matched_frags), len(chemicals)))

#     # ms1 peaks that are also fragmented
#     positives = []
#     for ms1_peak in matches_fullscan:
#         frag_peak = matches_fullscan[ms1_peak]
#         frag_events = chem_to_frag_events[frag_peak]
#         if len(frag_events) > 0:
#             positives.append(frag_peak)

#     # fragmentation peaks that are not in ms1 peaks
#     negatives = []
#     for frag_peak in chemicals:
#         if frag_peak not in matched_frags:
#             frag_events = chem_to_frag_events[frag_peak]
#             if len(frag_events) > 0:
#                 negatives.append(frag_peak)

#     positives_count = get_chem_frag_counts(positives, chem_to_frag_events, min_ms1_intensity)
#     negatives_count = get_chem_frag_counts(negatives, chem_to_frag_events, min_ms1_intensity)

#     # peaks from ground truth (found in full-scan files) that are fragmented above the minimum intensity threshold
#     tp = [chem for chem in positives if positives_count[chem]['good'] > 0 and positives_count[chem]['bad'] == 0]
#     tp = len(tp)

#     # peaks from ground truth that are not fragmented + peaks from ground truth that are fragmented below the minimum intensity threshold.
#     fp = len(detected_ms1) - tp

#     # peaks not from ground truth that are fragmented above the minimum intensity threshold.
#     fn = [chem for chem in negatives if negatives_count[chem]['good'] > 0 and negatives_count[chem]['bad'] == 0]
#     fn = len(fn)

#     prec, rec, f1 = compute_pref_rec_f1(tp, fp, fn)
#     return tp, fp, fn, prec, rec, f1


def compute_performance_scenario_2(controller, dataset, min_ms1_intensity,
                                   fullscan_filename, fragfile_filename,
                                   fullscan_peaks_df, fragmentation_peaks_df,
                                   matching_mz_tol, matching_rt_tol,
                                   chem_to_frag_events=None):
    if chem_to_frag_events is None:  # read MS2 fragmentation events from pickled controller
        chem_to_frag_events = get_frag_events(controller, 2)

    # load the list of xcms-picked peaks
    detected_from_fullscan = df_to_chemicals(fullscan_peaks_df, fullscan_filename)
    detected_from_fragfile = df_to_chemicals(fragmentation_peaks_df, fragfile_filename)

    # match with xcms peak-picked ms1 data from fullscan file
    matches_fullscan = match(dataset, detected_from_fullscan, matching_mz_tol, matching_rt_tol, verbose=False)

    # match with xcms peak-picked ms1 data from fragmentation file
    matches_fragfile = match(dataset, detected_from_fragfile, matching_mz_tol, matching_rt_tol, verbose=False)

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

    prec, rec, f1 = compute_pref_rec_f1(tp, fp, fn)
    return tp, fp, fn, prec, rec, f1


def get_frag_events(controller, ms_level):
    '''
    Gets the fragmentation events for all chemicals for an ms level from the controller
    :param controller: A Top-N controller object
    :param ms_level: The MS-level (usually 2)
    :return: A dictionary where keys are chemicals and values are a list of fragmentation events
    '''
    filtered_frag_events = list(
        filter(lambda x: x.ms_level == ms_level, controller.environment.mass_spec.fragmentation_events))
    chem_to_frag_events = defaultdict(list)
    for frag_event in filtered_frag_events:
        key = frag_event.chem
        chem_to_frag_events[key].append(frag_event)
    return dict(chem_to_frag_events)


def count_frag_events(chem, chem_to_frag_events, min_ms1_intensity):
    '''
    Counts how many good and bad fragmentation events for each chemical (key).
    Good fragmentation events are defined as fragmentation events that occur when at the time of fragmentation,
    the chemical MS1 intensity is above the min_ms1_intensity threshold.
    :param chem: the chemical to count
    :param chem_to_frag_events: a dictionary of chemicals to frag events (from get_frag_events above())
    :return: a tuple of good and bad fragmentation event counts
    '''
    frag_events = chem_to_frag_events[chem]
    good_count = 0
    bad_count = 0
    for frag_event in frag_events:
        chem = frag_event.chem
        query_rt = frag_event.query_rt
        if get_absolute_intensity(chem, query_rt) < min_ms1_intensity:
            bad_count += 1
        else:
            good_count += 1
    return good_count, bad_count


def get_chem_frag_counts(chem_list, chem_to_frag_events, min_ms1_intensity):
    # get the count of good/bad fragmentation events for all chemicals in chem_list
    results = {}
    for i in range(len(chem_list)):
        chem = chem_list[i]
        try:
            good_count, bad_count = count_frag_events(chem, chem_to_frag_events, min_ms1_intensity)
        except KeyError:
            good_count = 0
            bad_count = 0
        results[chem] = {
            'good': good_count,
            'bad': bad_count
        }
    return results


def update_matched_status(dataset, matches_fullscan, matches_fragfile):
    '''
    Update a boolean flag in the Chemical object that tells us if it is found in fullscan or fragmentation data
    :param dataset: a list of Chemicals
    :param matches_fullscan: the result of matching Chemicals in dataset to fullscan file
    :param matches_fragfile: the result of matching Chemicals in dataset to fragmentation file
    :return: None, but the Chemical objects in dataset is modified
    '''
    found_in_fullscan = 0
    found_in_fragfile = 0
    for chem in dataset:
        if matches_fullscan is not None:  # check if a match is found in fullscan mzML
            if chem in matches_fullscan:
                chem.found_in_fullscan = True
                found_in_fullscan += 1
            else:
                chem.found_in_fullscan = False

        if matches_fragfile is not None:  # check if a match is found in fragmentation mzML
            if chem in matches_fragfile:
                chem.found_in_fragfile = True
                found_in_fragfile += 1
            else:
                chem.found_in_fragfile = False

    logger.info('Matched %d/%d in fullscan data, %d/%d in fragmentation data' % (found_in_fullscan, len(dataset),
                                                                                 found_in_fragfile, len(dataset)))


def compute_pref_rec_f1(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return prec, rec, f1


def calculate_performance(params):
    # get parameters
    fragfile = params['fragfile']
    N = params['N']
    rt_tol = params['rt_tol']
    roi_mz_tol = params['roi_mz_tol']
    roi_min_ms1_intensity = params['roi_min_ms1_intensity']
    fragmentation_min_ms1_intensity = params['fragmentation_min_ms1_intensity']
    min_rt = params['min_rt']
    max_rt = params['max_rt']
    roi_min_length = params['roi_min_length']
    fullscan_filename = params['fullscan_filename']
    P_peaks_df = params['P_peaks_df']
    Q_peaks_df = params['Q_peaks_df']
    matching_mz_tol = params['matching_mz_tol']
    matching_rt_tol = params['matching_rt_tol']
    scenario = params['scenario']

    controller_file = params['controller_file']
    chemicals_file = params['chemicals_file']

    if chemicals_file.endswith('.p'):
        logger.info('Loading chemicals')
        chemicals = load_obj(chemicals_file)
    else:
        logger.info('Extracting chemicals')
        chemicals = get_chemicals(chemicals_file, roi_mz_tol, roi_min_ms1_intensity, min_rt, max_rt,
                                  min_length=roi_min_length)

    if type(chemicals) == list:
        chemicals = np.array(chemicals)

    if controller_file.endswith('.p'):
        logger.info('Loading fragmentation events')
        controller = load_obj(controller_file)
        chem_to_frag_events = None
    else:
        logger.info('Extracting fragmentation events')
        controller = None
        precursor_df = get_precursor_info(controller_file)
        chem_to_frag_events = get_chem_to_frag_events(chemicals, precursor_df)

    # compute performance under each scenario
    logger.info('Computing performance under scenario %d' % scenario)
    tp, fp, fn, prec, rec, f1 = 0, 0, 0, 0, 0, 0
    if scenario == 1:
        tp, fp, fn, prec, rec, f1 = compute_performance_scenario_1(controller, chemicals,
                                                                   fragmentation_min_ms1_intensity,
                                                                   fullscan_filename, P_peaks_df,
                                                                   matching_mz_tol, matching_rt_tol,
                                                                   chem_to_frag_events=chem_to_frag_events)
    elif scenario == 2:
        fragfile_filename = os.path.basename(fragfile)
        tp, fp, fn, prec, rec, f1 = compute_performance_scenario_2(controller, chemicals,
                                                                   fragmentation_min_ms1_intensity,
                                                                   fullscan_filename, fragfile_filename,
                                                                   P_peaks_df, Q_peaks_df, matching_mz_tol,
                                                                   matching_rt_tol,
                                                                   chem_to_frag_events=chem_to_frag_events)

    return N, rt_tol, scenario, tp, fp, fn, prec, rec, f1


def get_chem_to_frag_events(chemicals, ms1_df):
    # used for searching later
    min_rts = np.array([min(chem.chromatogram.raw_rts) for chem in chemicals])
    max_rts = np.array([max(chem.chromatogram.raw_rts) for chem in chemicals])
    min_mzs = np.array([min(chem.chromatogram.raw_mzs) for chem in chemicals])
    max_mzs = np.array([max(chem.chromatogram.raw_mzs) for chem in chemicals])

    # loop over each fragmentation event in ms1_df, attempt to match it to chemicals
    chem_to_frag_events = defaultdict(list)
    for idx, row in ms1_df.iterrows():
        query_rt = row['ms1_scan_rt']
        query_mz = row['ms1_mz']
        query_intensity = row['ms1_intensity']
        scan_id = row['ms2_scan_id']

        chem = None
        # idx = _get_chem_indices(query_mz, query_rt, min_mzs, max_mzs, min_rts, max_rts)
        idx = None
        if len(idx) == 1:  # single match
            chem = chemicals[idx][0]

        elif len(
                idx) > 1:  # multiple matches, find the closest in intensity to query_intensity at the time of fragmentation
            matches = chemicals[idx]
            possible_intensities = np.array([get_absolute_intensity(chem, query_rt) for chem in matches])
            closest = find_nearest_index_in_array(possible_intensities, query_intensity)
            chem = matches[closest]

        # create frag event for the given chem
        if chem is not None:
            ms_level = 2
            peaks = []  # we don't know which ms2 peaks are linked to this chem object
            # key = get_key(chem)
            frag_event = ScanEvent(chem, query_rt, ms_level, peaks, scan_id)
            chem_to_frag_events[chem].append(frag_event)
    return dict(chem_to_frag_events)


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
    roi_params = RoiParams(mz_tol=mz_tol, min_length=min_length,
                              min_intensity=min_intensity, start_rt=start_rt, stop_rt=stop_rt)
    good_roi = make_roi(mzML_file, roi_params)

    # keep ROI that have at least one point above the minimum to fragment threshold
    keep = []
    for roi in good_roi:
        if np.count_nonzero(np.array(roi.intensity_list) > min_ms1_intensity) > 0:
            keep.append(roi)

    ps = None  # old_unused_experimental
    rtcc = RoiToChemicalCreator(ps, keep)
    chemicals = np.array(rtcc.chemicals)
    return chemicals


def evaluate_serial(all_params):
    results = []
    for params in all_params:
        res = calculate_performance(params)
        results.append(res)
        logger.info('N=%d rt_tol=%d scenario=%d tp=%d fp=%d fn=%d prec=%.3f rec=%.3f f1=%.3f\n' % res)
    result_df = pd.DataFrame(results, columns=['N', 'rt_tol', 'scenario', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1'])
    return result_df


def evaluate_parallel(all_params, pushed_dict=None):
    import ipyparallel as ipp
    rc = ipp.Client()
    dview = rc[:]  # use all engines​
    with dview.sync_imports():
        pass

    if pushed_dict is not None:
        dview.push(pushed_dict)

    results = dview.map_sync(calculate_performance, all_params)
    result_df = pd.DataFrame(results, columns=['N', 'rt_tol', 'scenario', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1'])
    return result_df


def get_absolute_intensity(chem, query_rt):
    return chem.max_intensity * chem.chromatogram.get_relative_intensity(query_rt - chem.rt)
