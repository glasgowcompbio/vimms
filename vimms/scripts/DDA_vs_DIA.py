import glob
import os

import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from tqdm.auto import tqdm

from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.mzmine import pick_peaks
from mass_spec_utils.library_matching.spec_libraries import SpectralLibrary

from vimms.Agent import TopNDEWAgent
from vimms.Box import BoxGrid
from vimms.BoxManager import BoxManager, BoxSplitter
from vimms.BoxVisualise import PlotPoints
from vimms.Common import load_obj, create_if_not_exist
from vimms.Controller import AgentBasedController, TopN_SmartRoiController, \
    TopNController, WeightedDEWController, AIF, SWATH, AdvancedParams
from vimms.Controller.box import IntensityNonOverlapController, \
    NonOverlapController
from vimms.Environment import Environment
from vimms.Evaluation import evaluate_multi_peak_roi_aligner
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiAligner

################################################################################
# Synthetic experiments
################################################################################


def run_experiment(result_folder, sample_list, controller_names, experiment_params, parallel=True):
    """
    Go through all the chemicals in result_folder, and run various controllers
    on it. Will try to run it in parallel using ipcluster, if it fails then do it serially.

    :param result_folder: the folder containing folders of chemicals in replicate
    :param sample_list: the list of sample names to describe each replicate of chemical
    :param controller_names: the list of controllers to run
    :param experiment_params: experimental parameters
    :return: None
    """
    pickle_files = glob.glob(os.path.join(result_folder, '*.p'))
    params_list = []
    for pf in pickle_files:
        parameters = {
            'pf': pf,
            'sample_list': sample_list,
            'result_folder': result_folder,
            'controller_names': controller_names,
            'experiment_params': experiment_params
        }
        params_list.append(parameters)

    run_serial = True
    if parallel:  # Try to run the controllers in parallel. If fails, then run it serially
        logger.warning('Running controllers in parallel, please wait ...')
        run_serial = False
        try:
            rc = ipp.Client()
            dview = rc[:]  # use all engines
            with dview.sync_imports():
                pass
            dview.map_sync(run_once, params_list)
        except OSError:  # cluster has not been started
            run_serial = True
            logger.warning('Failed: IPycluster not found')
        except ipp.error.TimeoutError:  # takes too long to run
            run_serial = True
            logger.warning('Failed: IPycluster time-out')

    if run_serial:  # if parallel is disabled, or any exception from above, run it serially
        logger.warning('Running controllers in serial mode, please wait ...')
        for parameters in params_list:
            run_once(parameters)


def run_once(parameters):
    """
    Run an experiment once
    :param parameters: experimental parameters
    :return: None
    """
    # get parameters
    pf = parameters['pf']
    sample_list = parameters['sample_list']
    result_folder = parameters['result_folder']
    controller_names = parameters['controller_names']
    experiment_params = parameters['experiment_params']
    pbar = False

    # get experiment name
    root, ext = os.path.splitext(pf)
    experiment_name = root.split(os.sep)[-1]
    logger.warning(pf)

    # get chemical list for the experiment
    chem_list = load_obj(pf)
    assert len(chem_list) == len(sample_list)

    # run a lot of controllers on the chemical list
    run_simulated_exp(result_folder, experiment_name, controller_names, sample_list, chem_list,
                      experiment_params, pbar)


def run_simulated_exp(result_folder, experiment_name, controller_names, sample_list, chem_list,
                      experiment_params, pbar):
    ionisation_mode = experiment_params['ionisation_mode']
    min_measure_rt = experiment_params['min_measure_rt']
    max_measure_rt = experiment_params['max_measure_rt']
    min_measure_mz = experiment_params['min_measure_mz']
    max_measure_mz = experiment_params['max_measure_mz']

    rt_box_size = experiment_params['rt_box_size']
    mz_box_size = experiment_params['mz_box_size']
    scan_duration_dict = experiment_params['scan_duration_dict']
    spike_noise = experiment_params['spike_noise']
    mz_noise = experiment_params['mz_noise']
    intensity_noise = experiment_params['intensity_noise']

    topN_params = experiment_params['topN_params']
    smartroi_params = experiment_params['smartroi_params']
    weighteddew_params = experiment_params['weighteddew_params']
    AIF_params = experiment_params['AIF_params']
    SWATH_params = experiment_params['SWATH_params']

    non_overlap_params = {**topN_params,
                          **experiment_params['non_overlap_params']}  # combine the two dicts
    intensity_non_overlap_params = {**topN_params, **experiment_params['non_overlap_params']}

    non_overlap_smartroi_params = {**non_overlap_params, **smartroi_params}
    intensity_non_overlap_smartroi_params = {**intensity_non_overlap_params, **smartroi_params}

    non_overlap_weighteddew_params = {**non_overlap_params, **weighteddew_params}
    intensity_non_overlap_weighteddew_params = {**intensity_non_overlap_params,
                                                **weighteddew_params}

    IE_topN_params = dict(topN_params)
    agent = TopNDEWAgent(**IE_topN_params)

    def make_grid():
        grid = BoxManager(
            box_geometry=BoxGrid(min_measure_rt, max_measure_rt, rt_box_size,
                                 0, 1500, mz_box_size)
        )
        return grid

    def make_intensity_grid():
        grid = BoxManager(
            box_geometry=BoxGrid(min_measure_rt, max_measure_rt, rt_box_size,
                                 0, 1500, mz_box_size),
            box_splitter=BoxSplitter(split=True)
        )
        return grid

    for controller_name in controller_names:

        grids = {
            'non_overlap': make_grid(),
            'non_overlap_smartroi': make_grid(),
            'non_overlap_weighteddew': make_grid(),
            'intensity_non_overlap': make_intensity_grid(),
            'intensity_non_overlap_smartroi': make_intensity_grid(),
            'intensity_non_overlap_weighteddew': make_intensity_grid()
        }

        for i, chems in enumerate(chem_list):
            params = AdvancedParams()
            params.default_ms1_scan_window = [min_measure_mz, max_measure_mz]

            controllers = {
                'topN': TopNController(advanced_params=params, **topN_params),
                'topN_exclusion': AgentBasedController(agent, advanced_params=params),
                'non_overlap': NonOverlapController(
                    grid=grids["non_overlap"],
                    advanced_params=params,
                    **non_overlap_params
                ),
                'non_overlap_smartroi': NonOverlapController(
                    grid=grids["non_overlap_smartroi"],
                    **non_overlap_smartroi_params
                ),
                'non_overlap_weighteddew': NonOverlapController(
                    grid=grids["non_overlap_weighteddew"],
                    advanced_params=params,
                    **non_overlap_weighteddew_params
                ),
                'intensity_non_overlap': IntensityNonOverlapController(
                    grid=grids["intensity_non_overlap"],
                    advanced_params=params,
                    **intensity_non_overlap_params
                ),
                'intensity_non_overlap_smartroi': IntensityNonOverlapController(
                    grid=grids["intensity_non_overlap_smartroi"],
                    advanced_params=params,
                    **intensity_non_overlap_smartroi_params
                ),
                'intensity_non_overlap_weighteddew': IntensityNonOverlapController(
                    grid=grids["intensity_non_overlap_weighteddew"],
                    advanced_params=params,
                    **intensity_non_overlap_weighteddew_params
                ),
                'smartroi': TopN_SmartRoiController(
                    non_overlap_smartroi_params['ionisation_mode'],
                    non_overlap_smartroi_params['isolation_width'],
                    non_overlap_smartroi_params['N'],
                    non_overlap_smartroi_params['mz_tol'],
                    non_overlap_smartroi_params['rt_tol'],
                    non_overlap_smartroi_params['min_ms1_intensity'],
                    non_overlap_smartroi_params['roi_params'],
                    non_overlap_smartroi_params['smartroi_params'],
                    min_roi_length_for_fragmentation=non_overlap_smartroi_params[
                        'min_roi_length_for_fragmentation'],
                    advanced_params=params
                ),
                'weighteddew': WeightedDEWController(
                    non_overlap_weighteddew_params['ionisation_mode'],
                    non_overlap_weighteddew_params['N'],
                    non_overlap_weighteddew_params['isolation_width'],
                    non_overlap_weighteddew_params['mz_tol'],
                    non_overlap_weighteddew_params['rt_tol'],
                    non_overlap_weighteddew_params['min_ms1_intensity'],
                    exclusion_t_0=non_overlap_weighteddew_params['exclusion_t_0'],
                    log_intensity=True,
                    advanced_params=params
                ),
                'AIF': AIF(advanced_params=params, **AIF_params),
                'SWATH': SWATH(advanced_params=params, **SWATH_params)
            }

            logger.warning('%s %s' % (sample_list[i], controller_name))
            output_folder = os.path.join(result_folder, controller_name, experiment_name)
            mzML_name = '%s_%s.mzML' % (experiment_name, sample_list[i])

            controller = controllers[controller_name]
            run_controller(min_measure_rt, max_measure_rt, ionisation_mode, chems, controller,
                           output_folder, mzML_name, pbar,
                           spike_noise, mz_noise, intensity_noise,
                           scan_duration_dict)


def run_controller(min_rt, max_rt, ionisation_mode, chems, controller,
                   out_dir, out_file, pbar,
                   spike_noise, mz_noise, intensity_noise,
                   scan_duration_dict):
    mass_spec = IndependentMassSpectrometer(ionisation_mode, chems,
                                            spike_noise=spike_noise,
                                            mz_noise=mz_noise,
                                            intensity_noise=intensity_noise,
                                            scan_duration=scan_duration_dict)
    env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=pbar,
                      out_dir=out_dir, out_file=out_file, save_eval=True, check_exists=True)
    env.run()


def mzml_to_boxname(mzml):
    front = os.path.dirname(mzml)
    back = os.path.basename(mzml).split(".")[0] + "_box.csv"
    box_file = os.path.join(front, back)
    return box_file


def count_boxes(box_file):
    with open(box_file, 'r') as f:
        return sum(ln.strip() != "" for ln in f) - 1


def get_experiment_mzmls(controller_name, out_dir, repeat):
    mzml_files = [f"{controller_name}_{i}.mzML" for i in range(repeat)]
    input_files = [os.path.join(out_dir, mzml_file) for mzml_file in mzml_files]
    return input_files


def multi_samples_eval(controller_name, out_dir, repeat, box_mzML, mzmine_template, mzmine_path):
    # if not a list, put it in a list
    if not isinstance(box_mzML, list):
        box_mzMLs = [box_mzML] * repeat
    else:
        box_mzMLs = box_mzML

    # run peak picking on each box mzML file
    mzmine_outs = []
    for box_mzML in box_mzMLs:
        peak_picking_outdir = os.path.dirname(os.path.abspath(box_mzML))
        pick_peaks(box_mzML, xml_template=mzmine_template, output_dir=peak_picking_outdir,
                   mzmine_command=mzmine_path,
                   force=False)
        seed_box_file = mzml_to_boxname(box_mzML)
        print('Found', count_boxes(seed_box_file), 'boxes in', box_mzML)
        mzmine_outs.append(seed_box_file)

    input_files = get_experiment_mzmls(controller_name, out_dir, repeat)
    samples = [f"{controller_name}_{i})" for i in range(repeat)]

    assert len(input_files) == repeat
    assert len(box_mzMLs) == repeat
    assert len(mzmine_outs) == repeat

    # create ROI aligner and call evaluation method
    aligner = RoiAligner(rt_tolerance=100)
    for i in range(repeat):
        aligner.add_picked_peaks(input_files[i], mzmine_outs[i], samples[i], 'mzmine',
                                 half_isolation_window=0.01)
    multi_eval = evaluate_multi_peak_roi_aligner(aligner, samples)
    return multi_eval


def successes(results):
    return [np.sum(r) for r in results["cumulative_coverage"]]


def results_to_df(all_results):
    data = []
    for i in range(len(all_results)):
        eval_res = all_results[i]
        for controller_name in eval_res:
            results = eval_res[controller_name]

            try:
                coverages = results["cumulative_coverage_prop"]
            except KeyError:
                coverages = results['cumulative_coverage_proportion']

            try:
                intensity_proportions = results["cumulative_coverage_intensities_prop"]
            except KeyError:
                intensity_proportions = results["cumulative_intensity_proportion"]

            assert len(coverages) == len(intensity_proportions)
            for j in range(len(coverages)):
                cov = coverages[j]
                intensity = intensity_proportions[j]
                row = (i, controller_name, j, cov, intensity)
                data.append(row)

    df = pd.DataFrame(data, columns=['repeat', 'controller', 'sample_num', 'coverage_prop',
                                     'intensity_prop'])
    return df


def plot_results(controller_names, eval_res, suptitle=None, outfile=None, cumulative=True):
    sns.set_context('poster')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for exp_name in controller_names:
        results = eval_res[exp_name]
        if cumulative:
            coverages = results['cumulative_coverage_proportion']
            intensity_proportions = results["cumulative_intensity_proportion"]
        else:
            coverages = results['coverage_proportion']
            intensity_proportions = results["intensity_proportion"]

        xis = list(range(1, len(coverages) + 1))

        if cumulative:
            ax1.set(xlabel="Num. Runs", ylabel="Cumulative Coverage Proportion",
                    title="Multi-Sample Cumulative Coverage")
        else:
            ax1.set(xlabel="Num. Runs", ylabel="Coverage Proportion",
                    title="Multi-Sample Coverage")

        ax1.plot(xis, coverages, label=exp_name)
        ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.05))

        if cumulative:
            ax2.set(xlabel="Num. Runs", ylabel="Cumulative Intensity Proportion",
                    title="Multi-Sample Cumulative Intensity Proportion")
        else:
            ax2.set(xlabel="Num. Runs", ylabel="Intensity Proportion",
                    title="Multi-Sample Intensity Proportion")

        ax2.plot(xis, intensity_proportions, label=exp_name)
        ax2.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
    fig.set_size_inches(18.5, 10.5)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=32)
    if outfile is not None:
        create_if_not_exist(os.path.dirname(outfile))
        plt.savefig(outfile, facecolor='white', transparent=False)


def print_results(controller_names, eval_res):
    for exp_name in controller_names:
        print(exp_name)
        print(f"Cumulative Coverage: {successes(eval_res[exp_name])}")
        print(f"Cumulative Coverage Proportion: \
            {eval_res[exp_name]['cumulative_coverage_prop']}")
        print(
            f"Cumulative Intensity Proportion: \
                {eval_res[exp_name]['cumulative_coverage_intensities_prop']}")
        print()


def to_eval_res_df(eval_res_list, controller_names, group_values, group_label,
                   sample_values, sample_label, cumulative=False):
    assert len(group_values) == len(eval_res_list)

    data = []
    for i in range(len(eval_res_list)):
        group_value = group_values[i]
        eval_res = eval_res_list[i]

        for controller_name in controller_names:
            results = eval_res[controller_name]
            if cumulative:
                coverages = results['cumulative_coverage_proportion']
                intensity_proportions = results["cumulative_intensity_proportion"]
            else:
                coverages = results['coverage_proportion']
                intensity_proportions = results["intensity_proportion"]

            assert len(coverages) == len(sample_values)
            for j in range(len(coverages)):
                cov = coverages[j]
                intensity = intensity_proportions[j]
                sample_value = sample_values[j]
                row = [group_value, sample_value, controller_name, cov, intensity]
                data.append(row)
    df = pd.DataFrame(data, columns=[group_label, sample_label, 'controller', 'coverage_prop',
                                     'intensity_prop'])
    return df


def plot_multi_results(df, x, suptitle=None, outfile=None, plot_type='boxplot',
                       cumulative=False, palette=None):

    sns.set_context(context='poster', font_scale=1, rc=None)
    figsize = (20, 10)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if plot_type == 'boxplot':
        sns.boxplot(x=x, y='coverage_prop', hue='controller', data=df, ax=axes[0],
                    palette=palette)
        sns.boxplot(x=x, y='intensity_prop', hue='controller', data=df, ax=axes[1],
                    palette=palette)
    elif plot_type == 'lineplot':
        sns.lineplot(x=x, y='coverage_prop', hue='controller', data=df, ax=axes[0],
                     palette=palette, err_style="bars", ci='sd')
        sns.lineplot(x=x, y='intensity_prop', hue='controller', data=df, ax=axes[1],
                     palette=palette, err_style="bars", ci='sd')
    else:
        raise ValueError('Invalid plot_type, must be boxplot or lineplot')

    if cumulative:
        axes[0].set_title('Cumulative Coverage')
        axes[1].set_title('Cumulative Intensity Proportion')
    else:
        axes[0].set_title('Coverage')
        axes[1].set_title('Intensity Proportion')

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=32)

    # axes[0].set_ylim([-0.10, 1.1])
    # axes[1].set_ylim([-0.10, 1.1])
    axes[1].get_legend().remove()
    plt.tight_layout()

    if outfile is not None:
        create_if_not_exist(os.path.dirname(outfile))
        plt.savefig(outfile, facecolor='white', transparent=False)


def plot_frag_events(exp_name, out_dir, repeat):
    exp_paths = get_experiment_mzmls(exp_name, out_dir, repeat)
    mzmls = [MZMLFile(p) for p in exp_paths]
    if len(mzmls) == 1:
        mzml = mzmls[0]
        fig, ax = plt.subplots(len(mzmls), 1)
        pp = PlotPoints.from_mzml(mzml)
        pp.plot_ms2s(ax)
        ax.set(title=f"{exp_name} Fragmentation Events", xlabel="RT (Seconds)", ylabel="m/z")
        fig.set_size_inches(20, len(mzmls) * 4)
    else:
        fig, axes = plt.subplots(len(mzmls), 1)
        for i, (mzml, ax) in enumerate(zip(mzmls, axes)):
            pp = PlotPoints.from_mzml(mzml)
            pp.plot_ms2s(ax)
            ax.set(title=f"{exp_name} Run {i + 1} Fragmentation Events", xlabel="RT (Seconds)",
                   ylabel="m/z")
        fig.set_size_inches(20, len(mzmls) * 4)

    plt.suptitle(exp_name, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


################################################################################
# Real experiments
################################################################################

def compare_spectra(chem_library, base_folder, methods, matching_thresholds,
                    matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        spectra = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        for thresh in matching_thresholds:
            print(method, len(spectra), thresh)
            identified = []

            with tqdm(total=len(spectra)) as pbar:

                spec_identified = 0
                for i in range(len(spectra)):
                    spec = spectra[i]
                    hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                       matching_min_match_peaks, matching_ms1_tol, thresh)
                    if len(hits) > 0:
                        spec_identified += 1
                        identified.extend([hit[0] for hit in hits])
                    pbar.update(1)
                pbar.close()

                hit_count = len(np.unique(np.array(identified)))
                num_chem = len(spectra)  # FIXME: is this correct??!!
                hit_prop = hit_count / num_chem

                hit_prop = float(spec_identified) / num_chem

                row = [method, num_chem, thresh, hit_count, hit_prop]
                results.append(row)

    df = pd.DataFrame(results, columns=[
                      'method', 'num_peaks', 'matching_threshold', 'hit_count', 'hit_prop'])
    return df


def get_chemical_names(msp_file):
    names = []
    with open(msp_file) as f:
        for line in f:
            if line.startswith('Name'):
                tokens = line.split(':')
                chemical_name = tokens[1].strip()
                names.append(chemical_name)
    return set(names)


def load_alignment_df(msdial_file):
    try:
        df = pd.read_csv(msdial_file, sep='\t', skiprows=4,
                         index_col='Alignment ID')
    except ValueError:
        df = pd.read_csv(msdial_file, sep='\t', skiprows=0, index_col='PeakID')
    return df


def get_hits(filename, chemical_names, threshold=70):
    
    # load MSDIAL alignment results
    basename = os.path.basename(filename)
    if '_replicates' in basename:
        new_basename = basename.replace('_replicates', '')
        filename = os.path.join(os.path.dirname(filename), new_basename)

    df = load_alignment_df(filename)

    # filter by 'Reverse dot product' above threshold
    # filtered = df[df['Reverse dot product'] > threshold]

    # filter by 'Total score' above threshold
    filtered = df[df['Total score'] > threshold]

    # get unique hits in the filtered results
    try:
        hits = set(filtered['Metabolite name'].values)
    except KeyError:
        hits = set(filtered['Title'].values)

    # get the intersection between the chemical names and the hits
    return df, hits.intersection(chemical_names)


def load_hits(base_folder, controller_name, sample_list, chemical_names, combine_single_hits=False):
    replicate = 0

    # load single hits
    hit_counts = []
    unique_hits = set()
    for sample in sample_list:
        msdial_output = '%s_%s_%d.msdial' % (
            controller_name, sample, replicate)
        single_file = os.path.join(base_folder, controller_name, msdial_output)
        _, hits = get_hits(single_file, chemical_names)
        # print(sample, len(hits))
        hit_counts.append(len(hits))
        unique_hits.update(hits)

    # single_hits = max(hit_counts)
    single_hits = hit_counts[0]

    # load multi hits
    if not combine_single_hits:
        aligned_files = glob.glob(os.path.join(
            base_folder, controller_name, 'AlignResult*.msdial'))
        aligned_file = aligned_files[0]
        _, ms2dec_hits = get_hits(aligned_file, chemical_names)
        multi_hits = len(ms2dec_hits)
    else:
        multi_hits = len(unique_hits)
    # print(controller_name, multi_hits)

    # print()
    return single_hits, multi_hits


def get_ss_ms_df(all_controllers, msp_file, base_folder, sample_list):
    results = []
    chemical_names = get_chemical_names(msp_file)

    single_hits = []
    multi_hits = []
    for controller_name in all_controllers:
        # combine_single_hits = True if controller_name == 'topN_exclusion' else False
        combine_single_hits = True
        ss, ms = load_hits(base_folder, controller_name, sample_list, chemical_names,
                           combine_single_hits=combine_single_hits)
        results.append((controller_name, ss, 'single sample'))
        results.append((controller_name, ms, 'multiple samples'))

    return pd.DataFrame(results, columns=['method', 'hit', 'sample availability'])


def spectral_distribution(chem_library, base_folder, methods, matching_threshold,
                          matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        spectra = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        with tqdm(total=len(spectra)) as pbar:
            for spec in spectra:
                # returns a list containing (spectrum_id, sc, c)
                hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                   matching_min_match_peaks, matching_ms1_tol, matching_threshold)
                if len(hits) > 0:
                    max_item = max(hits, key=lambda item: item[1])  # 1 is sc
                    spectrum_id = max_item[0]
                    max_score = max_item[1]
                    if max_score > 0.0:
                        row = [method, spectrum_id, max_score]
                        results.append(row)
                pbar.update(1)
            pbar.close()

    df = pd.DataFrame(results, columns=['method', 'spectrum_id', 'score'])
    return df


def spec_records_to_library(spectra):
    # convert spectral records to spectral library for comparison
    chem_library = SpectralLibrary()
    chem_library.records = {spec.spectrum_id: spec for spec in spectra}
    chem_library.sorted_record_list = chem_library._dic2list()
    return chem_library


def pairwise_spectral_distribution(chem_library, base_folder, methods, matching_threshold,
    matching_method, matching_ms1_tol, matching_ms2_tol, matching_min_match_peaks):
    results = []
    methods = ['ground_truth'] + methods
    for method in methods:
        if method == 'ground_truth':

            # get spectra of chemicals for comparison
            spectra = list(chem_library.records.values())

        else:

            # get msdial results
            results_folder = os.path.join(base_folder, method)
            spectra = load_obj(os.path.join(
                results_folder, 'matched_spectra.p'))

            # chem_library is the same as spectra
            chem_library = spec_records_to_library(spectra)

        # compare spectra to library, which is the spectra themselves
        with tqdm(total=len(spectra)) as pbar:
            for spec in spectra:
                try:
                    # returns a list containing (spectrum_id, sc, c)
                    hits = chem_library.spectral_match(spec, matching_method, matching_ms2_tol,
                                                       matching_min_match_peaks, matching_ms1_tol,
                                                       matching_threshold)
                    if len(hits) == 1:
                        first = hits[0]
                        first_spectrum_id = first[0]
                        assert first_spectrum_id == spec.spectrum_id
                        max_score = 0.0
                    else:
                        max_score = 0.0
                        for hit in hits:
                            hit_spectrum_id = hit[0]
                            score = hit[1]
                            if hit_spectrum_id != spec.spectrum_id and score > max_score:
                                max_score = score
                    
                    if max_score > 0.0:
                        row = [method, spec.spectrum_id, max_score]
                        # print('spectrum_id', spec.spectrum_id, 'max_score', 
                        #     max_score, 'hits', hits)
                        # print()
                        results.append(row)
                except TypeError:
                    pass
                pbar.update(1)
            pbar.close()

    df = pd.DataFrame(results, columns=['method', 'spectrum_id', 'score'])
    return df