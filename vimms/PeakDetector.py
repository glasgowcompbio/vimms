import numpy as np
import pandas as pd
from loguru import logger
from vimms.Roi import Roi, make_roi
from vimms.PythonMzmine import get_base_scoring_df

QCB_MZML2CHEMS_DICT = {'min_ms1_intensity': 1.75E5,
                  'mz_tol': 2,
                  'mz_units':'ppm',
                  'min_length':1,
                  'min_intensity':0,
                  'start_rt':0,
                  'stop_rt':1560}


def get_rois(mzml, min_roi_length, mzml2chems_dict=QCB_MZML2CHEMS_DICT):
    good_roi, junk_roi = make_roi(mzml, mz_tol=mzml2chems_dict['mz_tol'], mz_units=mzml2chems_dict['mz_units'],
                              min_length=min_roi_length, min_intensity=mzml2chems_dict['min_intensity'],
                              start_rt=mzml2chems_dict['start_rt'], stop_rt=mzml2chems_dict['stop_rt'])
    return good_roi, junk_roi


def mzml2classificationdata(mzmls, mzml_picked_peaks_files, min_roi_length=5, mzml2chems_dict=QCB_MZML2CHEMS_DICT,
                            mz_slack=0.01, drift_window_lengths=[5], rt_peak_tol=2, include_status=True):
    rois = []
    for i in range(len(mzmls)):
        good_roi, junk_roi = get_rois(mzmls[i], min_roi_length, mzml2chems_dict)
        rois.extend(good_roi)
        picked_peaks = get_base_scoring_df(mzml_picked_peaks_files[i])
        df_new = rois2classificationdata2(good_roi, picked_peaks, mz_slack=mz_slack,
                                          drift_window_lengths=drift_window_lengths, rt_peak_tol=rt_peak_tol,
                                          include_status=include_status)
        if i == 0:
            df = df_new
        else:
            df = pd.concat([df,df_new])
    return df, rois


class get_prob_classifier(object):
    def __init__(self, mzmls, mzml_picked_peaks_files, min_roi_length=5, mzml2chems_dict=QCB_MZML2CHEMS_DICT,
                 mz_slack=0.01, roi_change_n=5, rt_peak_tol=2):
        self.roi_change_n = roi_change_n
        df, rois = mzml2classificationdata(mzmls, mzml_picked_peaks_files, min_roi_length, mzml2chems_dict,
                                           mz_slack, [roi_change_n], rt_peak_tol, include_status=True)
        df = df.dropna(thresh=2)
        base_classes = ['Decrease', 'Increase', 'Noise', 'Top']
        self.probabilities = []
        for i in range(int(max(df.iloc[:, 0]) + 1)):
            i_classes = df['rt_status'].iloc[np.where(df.iloc[:, 0] == i)[0]]
            probs = np.array([sum(i_classes == base) for base in base_classes]) / len(i_classes)
            self.probabilities.append(probs)

    def predict(self, value):
        return self.probabilities[value]


def calculate_window_change(intensities, drift_window_len):
    return sum((np.array(intensities)[-(drift_window_len-1):] - np.array(intensities)[-drift_window_len:-1]) > 0)


def find_possible_peaks(roi, picked_peaks, mz_slack):
    rt_check1 = (picked_peaks['rt min'] >= roi.rt_list[0]) & (roi.rt_list[-1] >= picked_peaks['rt min'])
    rt_check2 = (picked_peaks['rt max'] >= roi.rt_list[0]) & (roi.rt_list[-1] >= picked_peaks['rt max'])
    rt_check3 = (picked_peaks['rt min'] <= roi.rt_list[0]) & (picked_peaks['rt max'] >= roi.rt_list[-1])
    rt_check = rt_check1 | rt_check2 | rt_check3
    # logger.debug('rt len ' + len(rt_check))
    # logger.debug('rt check ' + rt_check)
    # plus and minus one is just slack for the initial check
    initial_mz_check = (picked_peaks['m/z max'] + 1 >= roi.get_mean_mz()) & (
                roi.get_mean_mz() >= picked_peaks['m/z min'] - 1)
    # logger.debug('mz len ' + len(initial_mz_check))
    # logger.debug('mz check ' + initial_mz_check)
    possible_peaks = np.where(np.logical_and(rt_check, initial_mz_check))[0]
    updated_possible_peaks = []
    for j in possible_peaks:
        peak = picked_peaks.iloc[j]
        check_peak = np.nonzero((peak['rt min'] < roi.rt_list) & (roi.rt_list < peak['rt max']))[0]
        mean_mz = np.mean(np.array(roi.mz_list)[check_peak])
        if peak['m/z min'] - mz_slack < mean_mz < peak['m/z max'] + mz_slack:
            updated_possible_peaks.append(j)
    return updated_possible_peaks

def rois2classificationdata2(rois, picked_peaks, mz_slack=0.01, drift_window_lengths = [5], rt_peak_tol=2,
                             include_status=True):
    roi_change_list = [[] for i in range(len(drift_window_lengths))]
    rt_status_list = []
    for roi in rois:
        # get drift data
        for window in range(len(drift_window_lengths)):
            roi_change_list[window].extend([None for i in range(drift_window_lengths[window]-1)])
            roi_change = [calculate_window_change(roi.intensity_list[:i], drift_window_lengths[window])
                          for i in range(drift_window_lengths[window], roi.n+1)]
            roi_change_list[window].extend(roi_change)
        # get possible peaks
        if include_status:
            possible_peaks = find_possible_peaks(roi, picked_peaks, mz_slack)
            possible_peaks_list = picked_peaks.iloc[possible_peaks]
            # get data
            if not possible_peaks:
                rt_status_list.extend([0 for rt in roi.rt_list])
            else:
                for rt in roi.rt_list:
                    rt_status = 0
                    for j in range(len(possible_peaks_list.index)):
                        if possible_peaks_list['rt centre'].iloc[j] - rt_peak_tol <= rt <= possible_peaks_list['rt centre'].iloc[j] + rt_peak_tol:
                            rt_status = max(3, rt_status)
                        elif possible_peaks_list['rt min'].iloc[j] <= rt <= possible_peaks_list['rt centre'].iloc[j]:
                            rt_status = max(2, rt_status)
                        elif possible_peaks_list['rt centre'].iloc[j] <= rt <= possible_peaks_list['rt max'].iloc[j]:
                            rt_status = max(1, rt_status)
                        else:
                            rt_status = max(0, rt_status)
                    rt_status_list.append(rt_status)
    # convert rt status to classes
    if include_status:
        rt_status_list = np.array(rt_status_list)
        rt_status_list_str = np.array(['Unknown' for i in range(len(rt_status_list))], dtype="<U10")
        rt_status_list_str[np.where(rt_status_list == 0)[0]] = 'Noise'
        rt_status_list_str[np.where(rt_status_list == 1)[0]] = 'Decrease'
        rt_status_list_str[np.where(rt_status_list == 2)[0]] = 'Increase'
        rt_status_list_str[np.where(rt_status_list == 3)[0]] = 'Top'
    # save as data frame
    df = pd.DataFrame()
    for window in range(len(drift_window_lengths)):
        df['roi_change_' + str(drift_window_lengths[window])] = roi_change_list[window]
    if include_status:
        df['rt_status'] = rt_status_list_str
    return df

# def get_intensity_difference(roi_intensities, n, positive=True):
#     # add exception for short roi
#     difference = []
#     for i in range(len(roi_intensities) - n):
#         difference.append(np.log(roi_intensities[i + n]) - np.log(roi_intensities[i]))
#     if positive:
#         return max(difference)
#     else:
#         return min(difference)
#
#
# def get_max_increasing(roi_intensities, n_skip=0, increasing_TF=True):
#     # add exception for short roi
#     max_increasing = 0
#     for i in range(len(roi_intensities)):
#         current_increasing = 0
#         current_skip = 0
#         if len(roi_intensities[i:]) <= max_increasing:
#             break
#         for j in range(1, len(roi_intensities[i:])):
#             if (roi_intensities[i:][j] > roi_intensities[i:][j - 1 - current_skip]) == increasing_TF:
#                 current_increasing += 1 + current_skip
#                 current_skip = 0
#             else:
#                 current_skip += 1
#                 if current_skip > n_skip:
#                     max_increasing = max(max_increasing, current_increasing)
#                     break
#     return max_increasing
#
#
# def get_intensity_list(roi, max_length):
#     if max_length is None:
#         return roi.intensity_list
#     else:
#         return roi.intensity_list[0:max_length]

# def rois2classificationdata(rois, picked_peaks, mz_slack=0.01):
#     base_roi = []
#     base_status = []
#     split_roi = []
#     split_status = []
#     for roi in rois:
#         rt_check1 = (picked_peaks['rt min'] >= roi.rt_list[0]) & (roi.rt_list[-1] >= picked_peaks['rt min'])
#         rt_check2 = (picked_peaks['rt max'] >= roi.rt_list[0]) & (roi.rt_list[-1] >= picked_peaks['rt max'])
#         rt_check3 = (picked_peaks['rt min'] <= roi.rt_list[0]) & (picked_peaks['rt max'] >= roi.rt_list[-1])
#         rt_check = rt_check1 | rt_check2 | rt_check3
#         # plus and minus one is just slack for the initial check
#         initial_mz_check = (picked_peaks['m/z max'] + 1 >= roi.get_mean_mz()) & (
#                     roi.get_mean_mz() >= picked_peaks['m/z min'] - 1)
#         possible_peaks = np.nonzero(rt_check & initial_mz_check)[0]
#         if len(possible_peaks) == 0:
#             base_roi.append(roi)
#             split_roi.append(roi)
#             base_status.append(0)
#             split_status.append(0)
#         else:
#             updated_possible_peaks = []
#             for j in possible_peaks:
#                 peak = picked_peaks.iloc[j]
#                 check_peak = np.nonzero((peak['rt min'] < roi.rt_list) & (roi.rt_list < peak['rt max']))[0]
#                 mean_mz = np.mean(np.array(roi.mz_list)[check_peak])
#                 if peak['m/z min'] - mz_slack < mean_mz < peak['m/z max'] + mz_slack:
#                     updated_possible_peaks.append(j)
#             if len(updated_possible_peaks) == 0:
#                 base_roi.append(roi)
#                 split_roi.append(roi)
#                 base_status.append(0)
#                 split_status.append(0)
#             else:
#                 if len(updated_possible_peaks) == 1:
#                     base_roi.append(roi)
#                     split_roi.append(roi)
#                     base_status.append(1)
#                     split_status.append(1)
#                 if len(updated_possible_peaks) > 1:
#                     base_roi.append(roi)
#                     base_status.append(1)
#                     df = picked_peaks.iloc[updated_possible_peaks]
#                     df = df.sort_values(by=['rt min'])
#                     splits = (np.array(df['rt min'][1:]) + np.array(df['rt max'][0:-1])) / 2
#                     splits = np.insert(np.insert(splits, 0, 0), len(splits) + 1, 2000)
#                     for j in range(len(splits) - 1):
#                         check_range1 = roi.rt_list > splits[j]
#                         check_range2 = roi.rt_list < splits[j + 1]
#                         mz = np.array(roi.mz_list)[np.nonzero(check_range1 & check_range2)[0]].tolist()
#                         rt = np.array(roi.rt_list)[np.nonzero(check_range1 & check_range2)[0]].tolist()
#                         intensity = np.array(roi.intensity_list)[np.nonzero(check_range1 & check_range2)].tolist()
#                         split_roi.append(Roi(mz, rt, intensity))
#                         split_status.append(1)
#     return base_roi, base_status, split_roi, split_status
#
#
# def get_roi_classification_params(rois,  roi_param_dict):
#     df = pd.DataFrame()
#     if roi_param_dict['include_log_max_intensity']:
#         df['log_max_intensity'] = np.log([roi.get_max_intensity() for roi in rois])
#     if roi_param_dict['include_log_intensity_difference']:
#         df['log_intensity_difference'] = np.log(df['log_max_intensity']) - np.log([roi.get_min_intensity() for roi in rois])
#     if roi_param_dict['consecutively_change_max'] > 0:
#         for i in range(roi_param_dict['consecutively_change_max']):
#             df['n_increase_' + str(i)] = [get_max_increasing(roi.intensity_list, i, True) for roi in rois]
#             df['n_decrease_' + str(i)] = [get_max_increasing(roi.intensity_list, i, False) for roi in rois]
#             df['n_interaction_' + str(i)] = df['n_increase_' + str(i)] * df['n_decrease_' + str(i)]
#     if roi_param_dict['intensity_change_max'] > 0:
#         for i in range(roi_param_dict['intensity_change_max']):
#             df['intensity_increase_' + str(i)] = [get_intensity_difference(roi.intensity_list, i+1, True) for roi in rois]
#             df['intensity_decrease_' + str(i)] = [get_intensity_difference(roi.intensity_list, i+1, False) for roi in rois]
#             df['intensity_interaction_' + str(i)] = df['intensity_increase_' + str(i)] * df['intensity_decrease_' + str(i)]
#     if roi_param_dict['lag_max'] > 0:
#         for i in range(roi_param_dict['lag_max']):
#             df['autocorrelation_' + str(i+1)] = [roi.get_autocorrelation(i+1) for roi in rois]
#     return df


