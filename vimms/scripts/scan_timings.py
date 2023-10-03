import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import argparse
import glob
import os

import numpy as np
import seaborn as sns
from loguru import logger
import pymzml
from scipy import interpolate
from sklearn.metrics import mean_squared_error


import pylab as plt
from mass_spec_utils.data_import.mzml import MZMLFile


def parse_args():
    parser = argparse.ArgumentParser(description='Create scan time plots')
    parser.add_argument('file_or_folder', type=str)
    parser.add_argument('--save_plots', dest='save_plots', action='store_true')
    args = parser.parse_args()
    return args


def process_mzML_files(file_or_folder):
    if os.path.isdir(file_or_folder):
        print("Extracting mzml from folder")
        file_list = glob.glob(os.path.join(file_or_folder, '*.mzML'))
    else:
        print("Processing", file_or_folder)
        file_list = [file_or_folder]
    mzml_file_objects = {}
    timings = {}
    for mzml_file in file_list:
        mzml_file_objects[mzml_file] = MZMLFile(mzml_file)
        timings[mzml_file] = get_times(mzml_file_objects[mzml_file])
    return timings


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


def plot_timings(args, timings):
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


def get_data(file_timings, file_name, level, remove_outliers=False):
    """Get data with potential outlier removal."""
    flat_d = {k: v[k] for k, v in file_timings.items()}

    try:
        rts, deltas = zip(*flat_d[file_name][level])
    except KeyError:
        return np.array([]), np.array([])

    rts = np.array(rts)
    deltas = np.array(deltas)

    if remove_outliers:
        rts, deltas = remove_data_outliers(rts, deltas)

    return rts, deltas


def remove_data_outliers(rts, deltas):
    """Remove outliers from the data based on IQR method."""
    q1, q3 = np.percentile(deltas, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (deltas >= lower_bound) & (deltas <= upper_bound)

    return rts[mask], deltas[mask]


def plot_deltas(file_timings, files, labels, plot_type='box', remove_outliers=False):
    """Generate specified type of plot for each level of data."""
    plot_types = ['box', 'violin', 'scatter']
    if plot_type not in plot_types:
        raise ValueError(f"Invalid plot_type. Expected one of: {plot_types}")

    levels = [(1, 1), (1, 2), (2, 1), (2, 2)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, level in zip(axs.flatten(), levels):
        data = [get_data(file_timings, file, level, remove_outliers) for file in files]

        if plot_type in ['box', 'violin']:
            deltas = [deltas for rts, deltas in data]
            plot_func = sns.boxplot if plot_type == 'box' else sns.violinplot
            plot_func(ax=ax, data=deltas)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)

            if plot_type == 'violin':
                for i, delta in enumerate(deltas):
                    # Ensure that delta is not empty
                    if delta.size > 0:
                        median_val = np.median(delta)

                        # Ensure that the median is not NaN
                        if not np.isnan(median_val):
                            ax.annotate(f"{median_val:.2f}",
                                        (i, median_val),
                                        xytext=(40, 40),  # move the annotation to the side
                                        textcoords='offset points',
                                        ha='center',
                                        va='center',
                                        fontsize=12,
                                        color='red',
                                        weight='bold',
                                        arrowprops=dict(arrowstyle="->", color='red'),
                                        bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

        else:  # 'scatter'
            for (rts, deltas), label in zip(data, labels):
                ax.scatter(rts, deltas, alpha=0.25, label=label, s=5)
            ax.legend()

        ax.set_title(f"Level: {level}")

    plt.tight_layout()
    plt.show()


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

    logger.debug('Number of ms1 scans = %d' % count_ms1_scans)
    logger.debug('Number of ms2 scans = %d' % count_ms2_scans)
    logger.debug('Total scans = %d' % (count_ms1_scans + count_ms2_scans))
    logger.debug('Number of selected precursors = %d' % count_selected_precursors)
    return np.array(mzs), np.array(rts), np.array(intensities), np.array(
        cumsum_ms1_scans), np.array(cumsum_ms2_scans)


def plot_num_scans(real_cumsum_ms1, real_cumsum_ms2, simulated_cumsum_ms1, simulated_cumsum_ms2,
                   out_file=None, show_plot=True):
    plt.figure(figsize=(10, 10))
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

    if show_plot:
        plt.show()

    plt.close()

def compute_similarity(real_cumsum, simulated_cumsum):
    # Interpolate to a common grid
    common_grid = np.linspace(0, 7900, 1000)  # you can adjust the number of points

    # Create the interpolation functions
    real_interpolator = interpolate.interp1d(real_cumsum[:, 0], real_cumsum[:, 1], fill_value="extrapolate")
    simulated_interpolator = interpolate.interp1d(simulated_cumsum[:, 0], simulated_cumsum[:, 1], fill_value="extrapolate")

    # Interpolate the data
    real_interpolated = real_interpolator(common_grid)
    simulated_interpolated = simulated_interpolator(common_grid)

    # Compute the mean squared error
    mse = mean_squared_error(real_interpolated, simulated_interpolated)

    return mse


def main():
    args = parse_args()
    timings = process_mzML_files(args.file_or_folder)
    plot_timings(args, timings)


if __name__ == '__main__':
    main()
