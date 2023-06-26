from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from ms_deisotope.deconvolution import deconvolute_peaks
from ms_deisotope.deconvolution.utils import prepare_peaklist
from tqdm import tqdm


def get_blocks(mz_file, sort_by_size=False):
    block_counter = 0
    block_sizes = defaultdict(list)

    for s in mz_file.scans:
        if s.ms_level == 1:
            block_counter += 1
        block_sizes[block_counter].append(s)

    if sort_by_size:
        block_sizes = sorted(block_sizes.items(), key=lambda x: len(x[1]), reverse=True)
    else:
        block_sizes = list(block_sizes.items())

    return block_sizes


def plot_peaks(scan, precursors, relative=False):
    # Extract mz and intensity values
    mz = [peak[0] for peak in scan.peaks]
    intensity = [peak[1] for peak in scan.peaks]

    # Convert to relative intensity if needed
    if relative:
        max_intensity = max(intensity)
        intensity = [i / max_intensity for i in intensity]

    # Create the plot
    plt.figure(figsize=(10, 6))
    for m, i in zip(mz, intensity):
        # Check if m is close to any value in precursors
        is_precursor = any(np.isclose(m, p, atol=1e-6) for p in precursors)
        color = 'red' if is_precursor else 'C0'  # 'C0' is the default matplotlib color
        if not is_precursor:  # If not a precursor, draw the line
            plt.vlines(m, 0, i, colors=color)

    for m, i in zip(mz, intensity):
        is_precursor = any(np.isclose(m, p, atol=1e-6) for p in precursors)
        if is_precursor:  # If a precursor, draw the line
            plt.vlines(m, 0, i, colors='red')

    plt.xlabel('m/z')
    plt.ylabel('Intensity' + (' (relative)' if relative else ''))
    plt.title(f'MS1 Peaks -- {scan.rt_in_seconds}s')
    plt.show()


def check_blocks(blocks):
    for block_id, scans in blocks:
        ms1_scan = scans[0]
        ms2_scans = scans[1:]
        precursors = [s.precursor_mz for s in ms2_scans]

        print('block_id', block_id)
        print('scans', scans)
        print('precursors', precursors, '(', len(precursors), ')')

        plot_peaks(ms1_scan, precursors)

        print()


def plot_num_ms2_scans(list_of_blocks, labels):
    # Determine the layout of the subplots
    num_plots = len(list_of_blocks)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6), sharex=False, sharey=True)
    axs = axs.ravel()  # Flatten the array of axes

    # Iterate over each list of blocks
    for i, (blocks, label) in enumerate(zip(list_of_blocks, labels)):
        # Prepare empty lists for the x and y values of the plot
        ms1_times = []
        num_ms2_scans = []

        for block_id, scans in blocks:
            ms1_scan = scans[0]
            ms2_scans = scans[1:]
            ms1_time = ms1_scan.rt_in_seconds
            num_ms2 = len(ms2_scans)

            # Append the values to the respective lists
            ms1_times.append(ms1_time)
            num_ms2_scans.append(num_ms2)

        # Add to the scatter plot
        axs[i].scatter(ms1_times, num_ms2_scans, s=1, alpha=0.2)
        axs[i].set_title(label)

    # Set the labels for the x and y axes
    for ax in axs:
        ax.set_xlabel('MS1 Scan Time (seconds)')
        ax.set_ylabel('Number of MS2 Scans')

    # Adjust the layout
    plt.tight_layout()
    plt.show()


def peaks_to_dataframe(peak_set, block_id=None, precursors=None):
    peaks_list = []

    for peak in peak_set:
        peak_dict = {
            "a_to_a2_ratio": peak.a_to_a2_ratio,
            "area": peak.area,
            "average_mass": peak.average_mass,
            "charge": peak.charge,
            "chosen_for_msms": peak.chosen_for_msms,
            "envelope": peak.envelope,
            "full_width_at_half_max": peak.full_width_at_half_max,
            "index": peak.index,
            "intensity": peak.intensity,
            "most_abundant_mass": peak.most_abundant_mass,
            "mz": peak.mz,
            "neutral_mass": peak.neutral_mass,
            "score": peak.score,
            "signal_to_noise": peak.signal_to_noise,
        }

        if block_id is not None:
            peak_dict['block_id'] = block_id

        if precursors is not None:
            is_precursor = any(np.isclose(peak.mz, p, atol=1e-6) for p in precursors)
            peak_dict['is_precursor'] = is_precursor

        peaks_list.append(peak_dict)

    df = pd.DataFrame(peaks_list)
    return df


def score_peaks_in_block(blocks, idx, should_plot=False):
    block = blocks[idx]
    block_id, scans = block
    ms1_scan = scans[0]
    ms2_scans = scans[1:]
    precursors = [s.precursor_mz for s in ms2_scans]
    assert len(ms2_scans) == len(precursors)

    # Run deconvolution
    # Extract mz and intensity values
    mzs = np.array([peak[0] for peak in ms1_scan.peaks])
    intensities = np.array([peak[1] for peak in ms1_scan.peaks])

    charge_range = (2, 6)
    pl = prepare_peaklist((mzs, intensities))
    ps = deconvolute_peaks(pl, charge_range=charge_range)
    df = peaks_to_dataframe(ps.peak_set.peaks, block_id=block_id, precursors=precursors)

    # optional plots
    if should_plot:
        plot_peaks(ms1_scan, precursors)
        plt.show()
        sns.histplot(df['score'], bins=30)  # plot score distribution
        plt.show()

    return df, precursors


def block_with_most_ms2_scans(blocks):
    max_ms2_scans = 0
    block_with_most_ms2 = None

    for block in blocks:
        block_id, scans = block
        ms2_scans = scans[1:]  # Get all MS2 scans

        if len(ms2_scans) > max_ms2_scans:
            max_ms2_scans = len(ms2_scans)
            block_with_most_ms2 = block

    return block_with_most_ms2


def deconvolute_blocks(blocks):
    scores_list = []
    times_list = []

    for block in tqdm(blocks, desc='Processing blocks'):
        block_id, scans = block
        ms1_scan = scans[0]
        ms2_scans = scans[1:]

        mzs = np.array([peak[0] for peak in ms1_scan.peaks])
        intensities = np.array([peak[1] for peak in ms1_scan.peaks])

        charge_range = (2, 6)
        pl = prepare_peaklist((mzs, intensities))
        ps = deconvolute_peaks(pl, charge_range=charge_range)
        df = peaks_to_dataframe(ps.peak_set.peaks)
        scores = df['score'].tolist()

        scores_list.append(scores)
        times_list.append(ms1_scan.rt_in_seconds)

    return scores_list, times_list


def remove_outliers(scores):
    # First quartile (Q1)
    Q1 = np.percentile(scores, 25, interpolation='midpoint')

    # Third quartile (Q3)
    Q3 = np.percentile(scores, 75, interpolation='midpoint')

    # Interquartile range (IQR)
    IQR = Q3 - Q1

    # Defining outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return [score for score in scores if lower_bound <= score <= upper_bound]


def score_distribution_blocks(scores_list, blocks):
    scores_with_ms2 = []
    scores_without_ms2 = []

    for idx, block in enumerate(blocks):
        block_id, scans = block
        ms2_scans = scans[1:]
        scores = scores_list[idx]

        if len(ms2_scans) > 0:
            scores_with_ms2.extend(scores)
        else:
            scores_without_ms2.extend(scores)

    # Remove outliers
    scores_with_ms2 = remove_outliers(scores_with_ms2)
    scores_without_ms2 = remove_outliers(scores_without_ms2)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

    sns.boxplot(scores_with_ms2, ax=axs[0])
    axs[0].set_title('Blocks with MS2 scans')

    sns.boxplot(scores_without_ms2, ax=axs[1])
    axs[1].set_title('Blocks without MS2 scans')

    plt.show()


def estimate_baseline(avg_scores, window_size=100, quantile=0.10):
    # Convert to pandas Series for convenience
    avg_scores_series = pd.Series(avg_scores)

    # Compute rolling median with the specified window size
    rolling_median = avg_scores_series.rolling(window_size, center=True).median()

    # Compute lower quantile
    baseline = rolling_median.quantile(quantile)

    return baseline


def plot_average_scores(scores_list, times_list):
    avg_scores = [sum(scores) / len(scores) if len(scores) > 0 else 0 for scores in scores_list]

    # Estimate baseline
    baseline = estimate_baseline(avg_scores)

    # Plot average scores over time
    plt.figure(figsize=(10, 5))
    plt.plot(times_list, avg_scores, label='Average Score')
    plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('Average Score')
    plt.title('Average Score over Time')
    plt.legend()
    plt.show()

    # Filter out scores below the baseline
    filtered_scores = [score for score in avg_scores if score > baseline]
    # return baseline, filtered_scores
