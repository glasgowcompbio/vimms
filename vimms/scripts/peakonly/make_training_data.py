# make_training_data.py
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

HOME = str(Path.home())
DEFAULT_PEAKONLY_PATH = os.path.join(HOME, 'peakonly-master')


def load_json(json_file):
    with open(json_file, 'r') as f:
        payload = json.loads(f.read())
    return payload


def load_all_json(json_file_list):
    roi_dict = {'no_peaks': [],
                'peaks': []}  # sort by presence / absence of peaks
    for json_file in tqdm(json_file_list):
        payload = load_json(json_file)
        number_of_peaks = payload['number of peaks']
        if number_of_peaks == 0:
            roi_dict['no_peaks'].append(payload)
        else:
            roi_dict['peaks'].append(payload)
    return roi_dict


def extract_example(roi, min_length, max_length, label):
    if label:
        assert roi['number of peaks'] > 0
    # find a possible end-point
    end_points = list(range(len(roi['intensity'])))
    if roi['number of peaks'] > 0 and not label:
        # it is an roi with peaks in but we want a non-peak example
        # remove the points within peaks from the end_points
        for peak_start, peak_stop in roi['borders']:
            end_points = list(
                filter(lambda x: x < peak_start or x >= peak_stop, end_points))
    if label:
        # remove end points that are *not* in a peak
        for peak_start, peak_stop in roi['borders']:
            end_points = list(
                filter(lambda x: x >= peak_start and x < peak_stop,
                       end_points))
            # remove points that would leave something too short
    end_points = list(filter(lambda x: x >= min_length, end_points))
    # choose a random point
    if len(end_points) > 0:
        end_point = np.random.choice(end_points)
        # choose a length
        max_length = min(max_length, end_point)
        length = np.random.choice(range(min_length, max_length + 1))
        intensity_vals = roi['intensity'][
            end_point - (length - 1):end_point + 1]
        mz_vals = roi['mz'][end_point - (length - 1):end_point + 1]
        return mz_vals, intensity_vals
    else:
        return [], []


def sample_once(roi_dict, key, label, choice_list, args):
    mz_vals = []
    while len(mz_vals) == 0:
        roi_idx = np.random.choice(choice_list)
        mz_vals, intensity_vals = extract_example(roi_dict[key][roi_idx],
                                                  args.min_length,
                                                  args.max_length, label)
        if len(mz_vals) > 0 and not args.with_replacement:
            set(choice_list).remove(roi_idx)
            choice_list = list(choice_list)
    return {'label': label, 'mz_vals': mz_vals,
            'intensity_vals': intensity_vals}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create training data')
    parser.add_argument('--peakonly_path', dest='peakonly_path',
                        default=DEFAULT_PEAKONLY_PATH, type=str)
    parser.add_argument('--n_pos', dest='n_pos', default=500, type=int)
    parser.add_argument('--n_neg', dest='n_neg', default=500, type=int)
    parser.add_argument('--with_replacement', dest='with_replacement',
                        action='store_true')
    parser.add_argument('--min_length', dest='min_length', type=int, default=5)
    parser.add_argument('--max_length', dest='max_length',
                        type=int, default=50)
    parser.add_argument('--output_file_name', dest='output_file_name',
                        type=str, default='training_items.json')
    args = parser.parse_args()

    print("ARGUMENTS")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # check that the peakonly training data is downloaded
    data_root = os.path.join(args.peakonly_path, 'data', 'annotation')
    if not os.path.isdir(data_root):
        print(
            "Error: folder {} does not exist. Ensure you've downloaded "
            "peakonly annotated data".format(
                data_root))
        sys.exit(0)

    print("Successfully found peakonly annotated data")

    # collect all the .json files
    json_files = []
    data_folders = glob.glob(os.path.join(data_root, 'Original*'))
    print(data_folders)
    for data_folder in data_folders:
        json_files += glob.glob(os.path.join(data_folder, '*.json'))
    print("Collected {} json files".format(len(json_files)))

    print("Loading")
    roi_dict = load_all_json(json_files)
    print("Loaded:")
    for key, l in roi_dict.items():
        print('\t', key, len(l))

    peaks_choice_list = list(range(len(roi_dict['peaks'])))
    no_peaks_choice_list = list(range(len(roi_dict['no_peaks'])))

    print("Sampling")
    sample_set = []
    for i in tqdm(range(args.n_pos)):
        new_example = sample_once(roi_dict, 'peaks', True, peaks_choice_list,
                                  args)
        sample_set.append(new_example)
    for i in tqdm(range(args.n_neg)):
        new_example = sample_once(roi_dict, 'no_peaks', False,
                                  no_peaks_choice_list, args)
        sample_set.append(new_example)

    print("Saving")
    with open(args.output_file_name, 'w') as f:
        f.write(json.dumps(sample_set))

    print("Done")
