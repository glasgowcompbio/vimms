# TopN on the HeLA data

import sys
sys.path.append('/home/joewandy/vimms')

import os
import numpy as np
import seaborn as sns
from tqdm import tqdm

from vimms.Common import set_log_level_warning, create_if_not_exist
from vimms.scripts.scan_timings import count_stuff, plot_num_scans, compute_similarity
from vimms.scripts.check_fragmented_ions import plot_num_ms2_scans, plot_histograms
from vimms.scripts.check_fragmented_ions import compare_histograms, BlockDeconvoluter, plot_heatmaps
from mass_spec_utils.data_import.mzml import MZMLFile

### Setup Parameters
seed_mzml_file = '/home/joewandy/data/HELA_20ng_1ul__sol_3.mzML'
rt_range = [(0, 7200)]
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]

real_input_file = seed_mzml_file
real_mzs, real_rts, real_intensities, real_cumsum_ms1, real_cumsum_ms2 = count_stuff(
    real_input_file, min_rt, max_rt)

max_blocks = int(1E6)
discard_first = True

hela_mzml_file = MZMLFile(real_input_file)
hela_bd = BlockDeconvoluter(hela_mzml_file, max_blocks=max_blocks, discard_first=discard_first)

# Check results mzML files
base_dir = '/datastore/joewandy/check_score_threshold_1E4_hela'
result_dir = os.path.abspath(os.path.join(base_dir, 'hela_results'))
plot_dir = os.path.abspath(os.path.join(base_dir, 'hela_plots'))
create_if_not_exist(plot_dir)

sns.set_context('poster')
set_log_level_warning()

charge = (2, 6)
labels = ['HeLA (true)', 'HeLA (simulated)']
show_plot = False

scores = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
penalty_factors = ['0.25', '0.50', '0.75', '1.0', '1.25', '1.50', '1.75', '2.0']

# scores = [120, 160]
# penalty_factors = ['1.0', '2.0']

# Initialize arrays to track values
rmse_ms1_array = np.zeros((len(scores), len(penalty_factors)))
rmse_ms2_array = np.zeros((len(scores), len(penalty_factors)))
sum_of_abs_diff_array = np.zeros((len(scores), len(penalty_factors)))

# Total number of combinations of scores and penalty factors
total_combinations = len(scores) * len(penalty_factors)

# Initialize tqdm
pbar = tqdm(total=total_combinations, desc="Processing combinations")

for i, score in enumerate(scores):
    for j, penalty in enumerate(penalty_factors):

        out_base = f'hela_1E4_{charge[0]}_{charge[1]}_{score}_{penalty}'
        simulated_input_file = os.path.join(result_dir, out_base, 'output.mzML')
        # print(simulated_input_file)

        cumsum_scans_out = os.path.join(plot_dir, f'{out_base}_cumsum_scans.png')
        num_ms2_scans_scatter_out = os.path.join(plot_dir, f'{out_base}_num_ms2_scans_scatter.png')
        num_ms2_scans_histogram_out = os.path.join(plot_dir, f'{out_base}_num_ms2_scans_histogram.png')

        if os.path.exists(simulated_input_file):

            simulated_mzs, simulated_rts, simulated_intensities, simulated_cumsum_ms1, simulated_cumsum_ms2 = count_stuff(
                simulated_input_file, min_rt, max_rt)

            plot_num_scans(real_cumsum_ms1, real_cumsum_ms2, simulated_cumsum_ms1, simulated_cumsum_ms2,
                           out_file=cumsum_scans_out, show_plot=show_plot)

            # compute RMSE of cumulative number of MS1 and MS2 scans
            rmse_ms1 = np.sqrt(compute_similarity(real_cumsum_ms1, simulated_cumsum_ms1))
            rmse_ms2 = np.sqrt(compute_similarity(real_cumsum_ms2, simulated_cumsum_ms2))

            simulated_mz_file = MZMLFile(simulated_input_file)
            simulated_bd = BlockDeconvoluter(simulated_mz_file, max_blocks=max_blocks, discard_first=discard_first)

            # compute sum of absolute difference of the two histograms of MS2 scans
            plot_num_ms2_scans(hela_bd, simulated_bd, labels, s=40, lo=0, hi=100,
                               out_file=num_ms2_scans_scatter_out, show_plot=show_plot)
            plot_histograms(hela_bd, simulated_bd, labels,
                            out_file=num_ms2_scans_histogram_out, show_plot=show_plot)
            sum_of_abs_diff = compare_histograms(hela_bd, simulated_bd)

            # Record the values
            rmse_ms1_array[i, j] = rmse_ms1
            rmse_ms2_array[i, j] = rmse_ms2
            sum_of_abs_diff_array[i, j] = sum_of_abs_diff

            # print(f'score={score} penalty={penalty} rmse_ms1={rmse_ms1} rmse_ms2={rmse_ms2} num_ms2_diff={sum_of_abs_diff}')

        else:
            print(f"The file {simulated_input_file} does not exist.")

        # Once done processing a combination, update the progress bar
        pbar.update()

# Close the progress bar once all combinations have been processed
pbar.close()
out_heatmap = os.path.join(plot_dir, 'heatmap.png')
plot_heatmaps(rmse_ms1_array, rmse_ms2_array, sum_of_abs_diff_array, scores, penalty_factors, out_file=out_heatmap, show_plot=show_plot)