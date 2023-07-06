from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ms_deisotope.deconvolution import deconvolute_peaks
from ms_deisotope.deconvolution.utils import prepare_peaklist
from tqdm import tqdm

ALL_BLOCKS = int(1E6)
ATOL = 0.01


def plot_num_ms2_scans(reference_block_deconvoluter, simulated_block_deconvoluter, labels,
                       s=3, alpha=1.0, lo=0, hi=int(1E6)):
    list_of_block_deconvoluters = [reference_block_deconvoluter, simulated_block_deconvoluter]

    # Determine the layout of the subplots
    num_plots = len(list_of_block_deconvoluters)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    # Create the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 6), sharex=True, sharey=True)
    axs = axs.ravel()  # Flatten the array of axes

    # Iterate over each list of blocks
    for i, (bd, label) in enumerate(zip(list_of_block_deconvoluters, labels)):
        # Prepare empty lists for the x and y values of the plot
        ms1_times = []
        num_ms2_scans = []

        for block_id, scans in bd.blocks:
            if lo <= block_id <= hi:
                ms1_scan = scans[0]
                ms2_scans = scans[1:]
                ms1_time = ms1_scan.rt_in_seconds
                num_ms2 = len(ms2_scans)

                # Append the values to the respective lists
                ms1_times.append(ms1_time)
                num_ms2_scans.append(num_ms2)

        axs[i].scatter(ms1_times, num_ms2_scans, s=s, alpha=alpha)
        axs[i].set_title(label)

    # Set the labels for the x and y axes
    for ax in axs:
        ax.set_xlabel('MS1 Scan Time (seconds)')
        ax.set_ylabel('Number of MS2 Scans')

    # Adjust the layout
    plt.tight_layout()
    plt.show()


class BlockDeconvoluter:
    def __init__(self, mz_file, max_blocks=ALL_BLOCKS, discard_first=False):
        self.blocks = self._get_blocks(mz_file, max_blocks=max_blocks, discard_first=discard_first)
        self._reset()

    def check_blocks(self, lo=0, hi=ALL_BLOCKS):
        for block in self.blocks[lo: hi]:
            self.plot_block(block)

    def plot_block(self, block):
        block_id, scans = block
        ms1_scan = scans[0]
        ms2_scans = scans[1:]
        precursors = [s.precursor_mz for s in ms2_scans]
        print('block_id', block_id)
        print('ms1_scan', ms1_scan.precursor_mz, '@', ms1_scan.rt_in_seconds)
        print('precursors', precursors, '(', len(precursors), ')')
        print('len(ms2_scans)', len(ms2_scans))
        self._plot_peaks(ms1_scan, precursors)
        print()

    def block_with_most_ms2_scans(self):
        max_ms2_scans = 0
        idx_found = None

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            block_id, scans = block
            ms2_scans = scans[1:]  # Get all MS2 scans

            if len(ms2_scans) > max_ms2_scans:
                max_ms2_scans = len(ms2_scans)
                idx_found = i

        print(idx_found)
        largest_block = self.blocks[idx_found]
        largest_df = self.dfs[idx_found]
        largest_precursors = self.precursor_list[idx_found]

        return largest_block, largest_df, largest_precursors

    def find_similar(self, to_find):
        to_find = to_find[1][0].rt_in_seconds

        prev_block = None
        curr_block = None
        block_after = None

        for block in self.blocks:
            prev_block = curr_block
            curr_block = block
            block_id, scans = block
            ms1_scan = scans[0]
            if ms1_scan.rt_in_seconds > to_find:
                block_after = curr_block
                break

        block_before = prev_block
        return block_before, block_after

    def deconvolute_blocks(self, decon_config=None):
        self._reset()

        for block in tqdm(self.blocks, desc='Processing blocks'):
            block_id, scans = block
            ms1_scan = scans[0]
            ms2_scans = scans[1:]
            precursors = [s.precursor_mz for s in ms2_scans]

            mzs = np.array([peak[0] for peak in ms1_scan.peaks])
            intensities = np.array([peak[1] for peak in ms1_scan.peaks])

            charge_range = (2, 6)
            pl = prepare_peaklist((mzs, intensities))
            ps = deconvolute_peaks(pl, decon_config=decon_config, charge_range=charge_range)
            df = self._peaks_to_dataframe(ps.peak_set.peaks, precursors=precursors)

            self.dfs.append(df)
            self.ms1_scans.append(ms1_scan)
            self.precursor_list.append(precursors)

            scores = df['score'].tolist() if not df.empty else []
            self.scores_list.append(scores)
            self.times_list.append(ms1_scan.rt_in_seconds)

    def score_distribution_blocks(self):
        scores_with_ms2 = []
        scores_without_ms2 = []

        for idx, block in enumerate(self.blocks):
            block_id, scans = block
            ms2_scans = scans[1:]
            scores = self.scores_list[idx]

            if len(ms2_scans) > 0:
                scores_with_ms2.extend(scores)
            else:
                scores_without_ms2.extend(scores)

        scores_with_ms2 = self._remove_outliers(scores_with_ms2)
        scores_without_ms2 = self._remove_outliers(scores_without_ms2)

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

        sns.boxplot(scores_with_ms2, ax=axs[0])
        axs[0].set_title('Blocks with MS2 scans')

        sns.boxplot(scores_without_ms2, ax=axs[1])
        axs[1].set_title('Blocks without MS2 scans')

        plt.show()

    def plot_average_scores(self, window_size=10):
        avg_scores = [sum(scores) / len(scores) if len(scores) > 0 else 0 for scores in
                      self.scores_list]
        num_peaks = [len(scores) if len(scores) > 0 else 0 for scores in
                     self.scores_list]

        baseline = self._estimate_baseline(avg_scores, window_size=window_size)
        print(baseline)

        plt.figure(figsize=(10, 5))
        plt.plot(self.times_list, avg_scores, label='Average Score',
                 marker='o')  # Added marker='o' for circular markers
        plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
        plt.xlabel('Time (in seconds)')
        plt.ylabel('Average Score')
        plt.title('Average Score over Time')
        plt.legend()
        plt.show()

        filtered_scores = [score for score in avg_scores if score > baseline]
        return filtered_scores

    def plot_minimum_scores(self):
        min_scores = [min(scores) if len(scores) > 0 else 0 for scores in self.scores_list]

        plt.figure(figsize=(10, 5))
        plt.plot(self.times_list, min_scores, label='Minimum Score',
                 marker='o')  # Added marker='o' for circular markers
        plt.xlabel('Time (in seconds)')
        plt.ylabel('Minimum Score')
        plt.title('Minimum Score over Time')
        plt.legend()
        plt.show()

    def plot_minimum_scores_fragmented(self, plot_type='line'):
        min_scores_fragmented = []
        signal_to_noise_ratios = []
        for df in self.dfs:
            if 'is_precursor' in df.columns:
                precursor_scores = df[df['is_precursor'] == True]['score']
                if not precursor_scores.empty:  # Added check for empty dataframe
                    min_score_idx = precursor_scores.idxmin()
                    min_score = df.loc[min_score_idx, 'score']
                    sn_ratio = df.loc[min_score_idx, 'signal_to_noise']
                    min_scores_fragmented.append(min_score)
                    signal_to_noise_ratios.append(int(sn_ratio))
                else:  # Handling case when precursor_scores is empty
                    min_scores_fragmented.append(None)
                    signal_to_noise_ratios.append(None)
            else:
                min_scores_fragmented.append(0)
                signal_to_noise_ratios.append(0)  # Assuming a default value of 0

        plt.figure(figsize=(10, 5))

        if plot_type == 'scatter':
            plt.scatter(min_scores_fragmented, signal_to_noise_ratios,
                        label='Signal-to-Noise Ratio vs Min Score', marker='o')
            plt.xlabel('Minimum Score')
            plt.ylabel('Signal-to-Noise Ratio')
        else:  # Default to line plot
            plt.plot(self.times_list, min_scores_fragmented, label='Minimum Score (fragmented)',
                     marker='o')
            for i, txt in enumerate(signal_to_noise_ratios):
                if txt is not None:  # Check to make sure there's something to annotate
                    plt.annotate(txt, (self.times_list[i], min_scores_fragmented[i]))
            plt.xlabel('Time (in seconds)')
            plt.ylabel('Minimum Score')

        plt.title('Minimum Score of Fragmented Peaks over Time')
        plt.legend()
        plt.show()

    def plot_maximum_scores(self):
        max_scores = [max(scores) if len(scores) > 0 else 0 for scores in self.scores_list]

        plt.figure(figsize=(10, 5))
        plt.plot(self.times_list, max_scores, label='Maximum Score',
                 marker='o')  # Added marker='o' for circular markers
        plt.xlabel('Time (in seconds)')
        plt.ylabel('Maximum Score')
        plt.title('Maximum Score over Time')
        plt.legend()
        plt.show()

    def plot_maximum_scores_fragmented(self, plot_type='line'):
        max_scores_fragmented = []
        signal_to_noise_ratios = []
        for df in self.dfs:
            if 'is_precursor' in df.columns:
                precursor_scores = df[df['is_precursor'] == True]['score']
                if not precursor_scores.empty:  # Added check for empty dataframe
                    max_score_idx = precursor_scores.idxmax()
                    max_score = df.loc[max_score_idx, 'score']
                    sn_ratio = df.loc[max_score_idx, 'signal_to_noise']
                    max_scores_fragmented.append(max_score)
                    signal_to_noise_ratios.append(int(sn_ratio))
                else:  # Handling case when precursor_scores is empty
                    max_scores_fragmented.append(None)
                    signal_to_noise_ratios.append(None)
            else:
                max_scores_fragmented.append(0)
                signal_to_noise_ratios.append(0)  # Assuming a default value of 0

        plt.figure(figsize=(10, 5))

        if plot_type == 'scatter':
            plt.scatter(max_scores_fragmented, signal_to_noise_ratios,
                        label='Signal-to-Noise Ratio vs Max Score', marker='o')
            plt.xlabel('Maximum Score')
            plt.ylabel('Signal-to-Noise Ratio')
        else:  # Default to line plot
            plt.plot(self.times_list, max_scores_fragmented, label='Maximum Score (fragmented)',
                     marker='o')
            for i, txt in enumerate(signal_to_noise_ratios):
                if txt is not None:  # Check to make sure there's something to annotate
                    plt.annotate(txt, (self.times_list[i], max_scores_fragmented[i]))
            plt.xlabel('Time (in seconds)')
            plt.ylabel('Maximum Score')

        plt.title('Maximum Score of Fragmented Peaks over Time')
        plt.legend()
        plt.show()

    def _reset(self):
        self.dfs = []
        self.ms1_scans = []
        self.precursor_list = []
        self.scores_list = []
        self.times_list = []

    def _get_blocks(self, mz_file, max_blocks=ALL_BLOCKS, discard_first=False):
        block_counter = 0
        block_sizes = defaultdict(list)

        for s in mz_file.scans:
            if s.ms_level == 1:
                block_counter += 1
                if block_counter >= max_blocks + 1:
                    break
            # If discard_first is True and we're on the first block, do nothing.
            # Otherwise, add the scan to the block.
            if not (discard_first and block_counter == 1):
                block_sizes[block_counter].append(s)

        block_sizes = list(block_sizes.items())
        return block_sizes

    def _plot_peaks(self, scan, precursors, relative=False):
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
            is_precursor = any(np.isclose(m, p, atol=ATOL) for p in precursors)
            # print(m, i, is_precursor)
            color = 'red' if is_precursor else 'C0'  # 'C0' is the default matplotlib color
            plt.vlines(m, 0, i, colors=color, zorder=2 if is_precursor else 1)

            # If a precursor, annotate the line
            if is_precursor:
                plt.annotate(f'm/z={m:.4f}', (m, i), textcoords="offset points", xytext=(0, 10),
                             ha='center', color='red')

        plt.xlabel('m/z')
        plt.ylabel('Intensity' + (' (relative)' if relative else ''))
        plt.title(f'MS1 Peaks -- {scan.rt_in_seconds}s')
        plt.show()

    def _peaks_to_dataframe(self, peak_set, block_id=None, precursors=None):
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

            is_precursor = False
            if precursors is not None:
                is_precursor = any(np.isclose(peak.mz, p, atol=ATOL) for p in precursors)
            peak_dict['is_precursor'] = is_precursor

            peaks_list.append(peak_dict)

        df = pd.DataFrame(peaks_list)
        return df

    def _remove_outliers(self, scores):
        Q1 = np.percentile(scores, 25, interpolation='midpoint')
        Q3 = np.percentile(scores, 75, interpolation='midpoint')
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        return [score for score in scores if lower_bound <= score <= upper_bound]

    def _estimate_baseline(self, avg_scores, window_size=10, quantile=0.10):
        avg_scores_series = pd.Series(avg_scores)
        rolling_median = avg_scores_series.rolling(window_size, center=True).median()
        baseline = rolling_median.quantile(quantile)

        return baseline
