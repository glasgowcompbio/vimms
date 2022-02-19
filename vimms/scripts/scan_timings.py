import argparse
import glob
import os
import sys

import pylab as plt
from mass_spec_utils.data_import.mzml import MZMLFile

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


def get_times(mzml_object):
    times = {(1, 1): [], (1, 2): [], (2, 1): [], (2, 2): []}
    for i, s in enumerate(mzml_object.scans[:-1]):
        if i == 0:
            continue  # skip the first one...
        current_scan = s
        next_scan = mzml_object.scans[i + 1]
        delta_t = next_scan.rt_in_seconds - current_scan.rt_in_seconds
        current_level = s.ms_level
        next_level = next_scan.ms_level
        times[(current_level, next_level)].append(
            (current_scan.rt_in_seconds, delta_t))
    to_remove = set()
    for key in times:
        if len(times[key]) == 0:
            to_remove.add(key)
    for key in to_remove:
        del times[key]
    return times


# flake8: noqa: C901
def main():
    global rt
    parser = argparse.ArgumentParser(description='Create scan time plots')
    parser.add_argument('file_or_folder', type=str)
    parser.add_argument('--save_plots', dest='save_plots', action='store_true')
    args = parser.parse_args()
    if os.path.isdir(args.file_or_folder):
        print("Extracting mzml from folder")
        file_list = glob.glob(os.path.join(args.file_or_folder, '*.mzML'))
    else:
        print("Individual file")
        file_list = [args.file_or_folder]
    mzml_file_objects = {}
    timings = {}
    for mzml_file in file_list:
        mzml_file_objects[mzml_file] = MZMLFile(mzml_file)
        timings[mzml_file] = get_times(mzml_file_objects[mzml_file])
    # plot
    for mo, t in timings.items():
        nsp = len(t)  # number of subplots
        plt.figure(figsize=(20, 8))
        pos = 1
        for k, v in t.items():
            title = mo.split(os.sep)[-1] + str(k)
            plt.subplot(2, nsp, pos)
            plt.title(title)
            try:
                rt, de = zip(*v)
            except Exception:
                print("No data for " + str(k))
            plt.hist(de)
            pos += 1
        for k, v in t.items():
            title = mo.split(os.sep)[-1] + str(k)
            plt.subplot(2, nsp, pos)
            plt.title(title)
            try:
                rt, de = zip(*v)
                plt.plot(rt, de, 'ro')
            except Exception:
                print("No data for " + str(k))
            pos += 1

        if args.save_plots:
            plot_filename = mo + '.png'
            plt.savefig(plot_filename)
        else:
            plt.show()


if __name__ == '__main__':
    main()
