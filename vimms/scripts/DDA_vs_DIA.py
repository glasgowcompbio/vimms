import glob
import os
from collections import defaultdict

import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.mzmine import pick_peaks
from mass_spec_utils.library_matching.spec_libraries import SpectralLibrary
from tqdm.auto import tqdm

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
from vimms.Evaluation import evaluate_multi_peak_roi_aligner, RealEvaluator
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

def evaluate_fragmentation(aligned_file, eval_using, sample_col_name,
                           sample_list, fragmentation_folder,
                           methods, replicates, isolation_width):
    """
    Evaluate boxes against fragmentation spectra using the `RealEvaluator` class.
    Args:
        aligned_file: Path to aligned CSV file containing boxes.
                      Can be produced by MzMine or MS-DIAL
        eval_using: Which tool produced the aligned file, either 'mzmine' or 'msdial'
        sample_col_name: a string to indicate which are the sample intensity columns
                         in the MS-DIAL results, e.g. 'beer'
        sample_list: a list of sample names, e.g. ['beer1', 'beer2', ...]
        fragmentation_folder: the folder containing fragmentation mzMLs
        methods: the name of methods, e.g. ['topN', 'topN_exclusion', ..]
        replicates: the number of replicates
        isolation_width: isolation width

    Returns: a dictionary where the key is method, and value is a RealEvaluator object.

    """
    eval_res = {}
    for method, replicate in zip(methods, replicates):
        print()
        print(method)

        assert eval_using in ['mzmine', 'msdial']
        if eval_using == 'mzmine':
            eva = RealEvaluator.from_aligned(aligned_file)
        elif eval_using == 'msdial':
            eva = RealEvaluator.from_aligned_msdial(aligned_file, sample_col_name)

        method_folder = os.path.join(fragmentation_folder, method)
        method_name = method.replace('_replicates', '')

        # TODO: check with Ross, but this seems incorrect?
        # this will create: [
        #     'fullscan_beer1_0': ['topN_beer1_0.mzML', 'topN_beer1_1.mzML', ...],
        #     'fullscan_beer2_0': ['topN_beer2_0.mzML', 'topN_beer2_1.mzML', ...],
        #     ...
        # ]
        # mzml_pairs = []
        # for sample in sample_list:
        #     fullscan_name = 'fullscan_%s_0' % sample
        #     mzmls = []
        #     for i in range(replicate):
        #         mzml = os.path.join(method_folder, '%s_%s_%d.mzML' % (method_name, sample, i))
        #         mzmls.append(mzml)
        #     pair = (fullscan_name, mzmls)
        #     mzml_pairs.append(pair)

        # TODO: check with Ross
        # this will create: [
        #     'fullscan_beer1_0': ['topN_beer1_0.mzML'],
        #     'fullscan_beer2_0': ['topN_beer2_0.mzML'],
        #     'fullscan_beer3_0': ['topN_beer3_0.mzML'],
        #     'fullscan_beer4_0': ['topN_beer4_0.mzML'],
        #     'fullscan_beer5_0': ['topN_beer5_0.mzML'],
        #     'fullscan_beer6_0': ['topN_beer6_0.mzML'],
        #     'fullscan_beer1_0': ['topN_beer1_1.mzML'],
        #     'fullscan_beer2_0': ['topN_beer2_1.mzML'],
        #     'fullscan_beer3_0': ['topN_beer3_1.mzML'],
        #     ...
        # ]
        mzml_pairs = []
        for i in range(replicate):
            for sample in sample_list:
                fullscan_name = 'fullscan_%s_0' % sample
                mzmls = [os.path.join(method_folder, '%s_%s_%d.mzML' % (method_name, sample, i))]
                pair = (fullscan_name, mzmls)
                mzml_pairs.append(pair)

        for fullscan_name, mzmls in mzml_pairs:
            print(fullscan_name, mzmls)
            eva.add_info(fullscan_name, mzmls, isolation_width=isolation_width)

        eval_res[method] = eva
    return eval_res


def print_evaluations(eval_res):
    for method in eval_res:
        eva = eval_res[method]
        print(method)
        print(eva.summarise())
        print()


def evas_to_reports(eval_res):
    reports = {method: eva.revaluation_report() for method, eva in eval_res.iteritems()}
    return reports


def eval_res_to_df(eval_res):
    dfs = []
    for method in eval_res:
        print(method)
        report = eval_res[method].evaluation_report()
        data = []

        metric_names = [
            'num_frags',
            'sum_cumulative_coverage',
            'cumulative_coverage_proportion',
            'cumulative_intensity_proportion'
        ]
        for metric_name in metric_names:
            for i, metric_value in enumerate(report[metric_name]):
                data.append((method, i, metric_value, metric_name))

        df = pd.DataFrame(data, columns=['method', 'sample_idx', 'metric_value', 'metric_name'])
        dfs.append(df)

    combined_df = pd.concat(dfs)
    return combined_df


def plot_coverage_intensity_props(df, selected_methods, suptitle=None):
    fig, axes = plt.subplots(1, 2, sharey=False, figsize=(20, 10))

    data = df[df['metric_name'] == 'cumulative_coverage_proportion']
    data = data[data['method'].isin(selected_methods)].reset_index(drop=True)
    g = sns.lineplot(data=data, x='sample_idx', y='metric_value', hue='method', ax=axes[0])

    g.set(ylabel='Coverage proportion')
    g.set(xlabel='Samples')
    axes[0].set_title('Coverage proportion vs samples')
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.74, -0.14))

    data = df[df['metric_name'] == 'cumulative_intensity_proportion']
    data = data[data['method'].isin(selected_methods)].reset_index(drop=True)
    g = sns.lineplot(data=data, x='sample_idx', y='metric_value', hue='method', ax=axes[1])

    g.set(ylabel='Intensity proportion')
    g.set(xlabel='Samples')
    axes[1].set_title('Intensity proportion vs samples')
    axes[1].get_legend().remove()

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=24)


def get_msdial_file(msdial_folder):
    """In a folder, get the filename containing MS-DIAL aligned peaklist.

    Args:
        msdial_folder: the folder containing MS-DIAL output

    Returns:
        the path to MS-DIAL aligned result in the folder
    """
    msdial_file = None

    # search for 'Height' file (CorrDec output)
    for filename in glob.glob(msdial_folder + '/*'):
        if 'Height' in filename and 'txt' in filename:
            msdial_file = filename
            break

    # if not found, search for 'ALignResult' file (ms2dec output)
    if msdial_file is None:
        for filename in glob.glob(msdial_folder + '/*'):
            if 'AlignResult' in filename and '.msdial' in filename:
                msdial_file = filename
                break

    return msdial_file


def eva_to_matches(method, eval_res, fullscan_spectra, allow_multiple=False):
    eva = eval_res[method]

    matches = defaultdict(list)
    for box, fullscan_peak in eva.box_to_fullscan_peaks.items():
        spectra = eva.box_to_frag_spectra[box]

        # there could be multiple spectra associated to each box
        # find spectra fragmented at the highest intensity and assign it to the fullscan peak
        if len(spectra) > 0:
            best_spectra = highest_intensity_at_frag(spectra)
            matches[fullscan_peak].append(best_spectra)
    matches = dict(matches)

    # filter by allow_multiple
    results = {}
    for fullscan_peak in eva.all_fullscan_peaks:
        spectra = []
        if fullscan_peak in matches:
            spectra = matches[fullscan_peak]

        if not allow_multiple and len(spectra) > 1:
            # choose highest intensity at fragmentaiton
            best_spectra = highest_intensity_at_frag(spectra)
            results[fullscan_peak] = [best_spectra]
        else:
            results[fullscan_peak] = spectra

    assert len(results) == len(fullscan_spectra)
    return results


def highest_intensity_at_frag(spectra):
    spectra_intensities = [spec.metadata['best_intensity_at_frag'] for spec in spectra]
    best_spectra = spectra[np.argmax(spectra_intensities)]
    return best_spectra


def match_spectra_list(spectra_1, spectra_2, mz_tol, rt_tol, allow_multiple=False):
    """
    Match two lists `spectra_1` and `spectra_2`. 
    Here `spectra_1` should be the fullscan features, while 
    `spectra_2` is the fragmentation features from MS-DIAL.

    - Each item in the list is a `SpectralRecord` object. This basically 
      corresponds to a 'box' (peak) from the aligned MS-DIAL results. 
      A box has average m/z and RT values (after alignment).
    - For each item in `spectra_1`, find the items in `spectra_2` 
      within mz_tol ppm and rt_tol seconds away.           

    Args:
        spectra_1: the first list of SpectralRecord objects
        spectra_2: the second list of SpectralRecord objects
        mz_tol: m/z tolerance in ppm
        rt_tol: RT tolerance in seconds
        allow_multiple: whether to allow multiple matches in the returned dictionary. 
        If False, select the one closest in m/z value (might not be the best thing to do).

    Returns:
        A dictionary of matches, where the key is each spectra in spectra_1, and values are
        spectra in spectra_2 that can be matched to spectra_1.
    """
    spectra_1 = np.array(spectra_1)
    spectra_2 = np.array(spectra_2)

    # create mz range for matching in ppm
    min_mzs = np.array([spec.precursor_mz * (1 - mz_tol / 1e6) for spec in spectra_2])
    max_mzs = np.array([spec.precursor_mz * (1 + mz_tol / 1e6) for spec in spectra_2])

    # create rt ranges for matching
    min_rts = np.array([spec.metadata['rt'] - rt_tol for spec in spectra_2])
    max_rts = np.array([spec.metadata['rt'] + rt_tol for spec in spectra_2])

    matches_dict = {}
    for query in spectra_1:  # loop over query and find a match
        matches = find_match(query, min_rts, max_rts, min_mzs, max_mzs,
                             spectra_2, allow_multiple)
        matches_dict[query] = matches
    return matches_dict


def find_match(query, min_rts, max_rts, min_mzs, max_mzs, spectra_arr, allow_multiple):
    # check ranges
    query_mz = query.precursor_mz
    query_rt = query.metadata['rt']
    min_rt_check = min_rts <= query_rt
    max_rt_check = query_rt <= max_rts
    min_mz_check = min_mzs <= query_mz
    max_mz_check = query_mz <= max_mzs
    idx = np.nonzero(min_rt_check & max_rt_check & min_mz_check & max_mz_check)[0]
    # print(idx)

    # get matching spectra
    if len(idx) == 0:  # no match
        return []

    elif len(idx) == 1:  # single match
        return [spectra_arr[idx][0]]

    else:  # multiple matches, take the closest in rt
        matches = spectra_arr[idx]
        if allow_multiple:
            return matches
        else:
            # pick the one closest in precursor m/z
            matches_mz = [spec.precursor_mz for spec in matches]
            diffs = [np.abs(mz - query_mz) for mz in matches_mz]
            idx = np.argmin(diffs)
            return [matches[idx]]


def compare_spectra(chem_library, base_folder, methods, matching_thresholds,
                    matching_method, matching_ms1_tol, matching_ms2_tol,
                    matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        matches = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        for thresh in matching_thresholds:
            print(method, thresh)
            annotated_compounds = []
            annotated_peaks = []
            total_compounds = len(chem_library.sorted_record_list)
            total_peaks = len(matches)

            with tqdm(total=total_peaks) as pbar:
                for k, v in matches.items():
                    for spec in v:
                        hits = chem_library.spectral_match(spec, matching_method,
                                                           matching_ms2_tol,
                                                           matching_min_match_peaks,
                                                           matching_ms1_tol, thresh)
                        if len(hits) > 0:
                            for item in hits:
                                spectrum_id = item[0]
                                score = item[1]
                                if score > 0.0:
                                    annotated_compounds.append(spectrum_id)
                                    annotated_peaks.append(k)
                    pbar.update(1)
                pbar.close()

            no_annotated_compounds = len(set(annotated_compounds))
            no_annotated_peaks = len(set(annotated_peaks))
            prop_annotated_peaks = no_annotated_peaks / total_peaks
            prop_annotated_compounds = no_annotated_compounds / total_compounds
            row = [
                method,
                thresh,
                no_annotated_compounds,
                no_annotated_peaks,
                prop_annotated_compounds,
                prop_annotated_peaks,
                set(annotated_peaks)
            ]
            results.append(row)

    df = pd.DataFrame(results, columns=[
        'method', 'matching_threshold',
        'no_annotated_compounds', 'no_annotated_peaks',
        'prop_annotated_compounds', 'prop_annotated_peaks',
        'annotated_peaks'])
    return df


def spectral_distribution(chem_library, base_folder, methods, matching_threshold,
                          matching_method, matching_ms1_tol, matching_ms2_tol,
                          matching_min_match_peaks):
    print('chem_library', chem_library)
    results = []
    for method in methods:
        print(method)

        # get msdial results
        results_folder = os.path.join(base_folder, method)
        spectra = load_obj(os.path.join(results_folder, 'matched_spectra.p'))

        # compare spectra
        with tqdm(total=len(spectra)) as pbar:
            for k, v in spectra.items():
                for spec in v:
                    # returns a list containing (spectrum_id, sc, c)
                    hits = chem_library.spectral_match(spec, matching_method,
                                                       matching_ms2_tol, matching_min_match_peaks,
                                                       matching_ms1_tol, matching_threshold)
                    if len(hits) > 0:
                        for item in hits:
                            spectrum_id = item[0]
                            score = item[1]
                            if score > 0.0:
                                row = [method, spectrum_id, score]
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


def pairwise_spectral_distribution(chem_library, base_folder, methods,
                                   matching_threshold, matching_method, matching_ms1_tol,
                                   matching_ms2_tol, matching_min_match_peaks):
    results = []
    methods = ['ground_truth'] + methods
    for method in methods:
        if method == 'ground_truth':

            # get spectra of chemicals for comparison
            spectra = list(chem_library.records.values())

        else:

            # get msdial results
            results_folder = os.path.join(base_folder, method)
            matched = load_obj(os.path.join(
                results_folder, 'matched_spectra.p'))
            spectra = []
            for k, v in matched.items():
                spectra.extend(v)

            # chem_library is the same as spectra
            chem_library = spec_records_to_library(spectra)

        # compare spectra to library, which is the spectra themselves
        with tqdm(total=len(spectra)) as pbar:
            for spec in spectra:
                # returns a list containing (spectrum_id, sc, c)
                hits = chem_library.spectral_match(spec, matching_method,
                                                   matching_ms2_tol, matching_min_match_peaks,
                                                   matching_ms1_tol, matching_threshold)
                if len(hits) > 0:
                    for item in hits:
                        spectrum_id = item[0]
                        score = item[1]
                        if score != 1.00:  # ignore matches to itself
                            row = [method, spectrum_id, score]
                            results.append(row)
                pbar.update(1)
            pbar.close()

    df = pd.DataFrame(results, columns=['method', 'spectrum_id', 'score'])
    return df


def matched_spectra_as_df(base_folder, methods):
    dfs = []
    score_dfs = []
    for method in methods:
        print(method)
        matched_spectra = load_obj(os.path.join(base_folder, method, 'matched_spectra.p'))

        # for each method, convert its matched_spectra to dataframe
        scores = []
        for k, v in matched_spectra.items():
            for spec in v:
                row = {
                    'method': method
                }
                row['precursor_mz'] = k.precursor_mz
                row.update((spec.metadata))
                if row['names'][0] == 'Unknown':
                    row['names'] = np.NaN
                else:
                    row['names'] = row['names'][0]
                scores.append(row)
        df = pd.DataFrame(scores)
        dfs.append(df)

        # filter df and extract scores only
        data = []
        for idx, row in df.iterrows():
            name = row['names']
            score = row['dot_product']
            if isinstance(name, str) and not np.isnan(score) and 'w/o MS2' not in name:
                row = [method, name, score]
                data.append(row)
        score_df = pd.DataFrame(data, columns=['method', 'name', 'score'])
        score_df = score_df.sort_values(
            'score', ascending=False).drop_duplicates('name').sort_index()
        score_dfs.append(score_df)

    df = pd.concat(dfs).reset_index(drop=True)
    score_df = pd.concat(score_dfs).reset_index(drop=True)
    return df, score_df


def plot_matching_thresholds_1(hit_prop_df, y='prop_annotated_peaks'):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(20, 5))

    selected_df = hit_prop_df.replace({
        'topN': 'Top-N',
        'topN_exclusion_replicates': 'Iterative\nExclusion',
        'intensity_non_overlap': 'Intensity\nNon-overlap'
    })
    selected_df = selected_df[selected_df['matching_threshold'].isin([0.2, 0.4, 0.6, 0.8])]

    methods = selected_df['method'].unique()
    colours = sns.color_palette(n_colors=len(methods))
    palette = {method: colour for method, colour in zip(methods, colours)}

    plot_df = selected_df[selected_df['method'].isin(['Top-N', 'SWATH', 'AIF'])]
    # line_at = plot_df[
    #     (plot_df['method'] == 'Top-N') &
    #     (plot_df['matching_threshold'] == 0.6)
    # ][y].values[0]
    g = sns.barplot(data=plot_df, x='matching_threshold',
                    hue='method', y=y, ax=axes[0], palette=palette, hue_order=[
            'AIF', 'SWATH', 'Top-N'])
    g.set(ylabel='Annotated features')
    g.set(xlabel='Matching threshold')
    sns.move_legend(g, "upper right", title='Method')
    axes[0].set_title('1 replicate')

    plot_df = selected_df[selected_df['method'].isin(
        ['topN_replicates', 'Iterative\nExclusion', 'Intensity\nNon-overlap'])]
    plot_df = plot_df.replace({'topN_replicates': 'Top-N', })
    g = sns.barplot(data=plot_df, x='matching_threshold',
                    hue='method', y=y, ax=axes[1], palette=palette, hue_order=[
            'Intensity\nNon-overlap', 'Top-N', 'Iterative\nExclusion'
        ])
    g.set(ylabel=None)
    g.set(xlabel='Matching threshold')
    sns.move_legend(g, "upper right", title='Method')
    axes[1].set_title('4 replicates')

    # axes[0].axhline(line_at, ls='--', color='k', alpha=0.60)
    # axes[1].axhline(line_at, ls='--', color='k', alpha=0.60)

    return palette


def plot_matching_thresholds_2(hit_prop_df, y='prop_annotated_peaks'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

    selected_df = hit_prop_df.replace({
        'topN': 'Top-N',
        'topN_exclusion_replicates': 'Iterative\nExclusion',
        'intensity_non_overlap': 'Intensity\nNon-overlap'
    })
    selected_df = selected_df[selected_df['matching_threshold'].isin(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]

    methods = selected_df['method'].unique()
    colours = sns.color_palette(n_colors=len(methods))
    palette = {method: colour for method, colour in zip(methods, colours)}

    plot_df = selected_df[selected_df['method'].isin(['Top-N', 'SWATH', 'AIF'])]
    g = sns.barplot(data=plot_df, x='matching_threshold', hue='method', hue_order=[
        'AIF', 'SWATH', 'Top-N'], y=y, ax=axes[0], palette=palette)
    g.set(ylabel='Annotated features')
    g.set(xlabel=None)
    sns.move_legend(g, "upper right", title='Method')
    axes[0].set_title('1 replicate')

    plot_df = selected_df[selected_df['method'].isin(
        ['topN_replicates', 'Iterative\nExclusion', 'Intensity\nNon-overlap'])]
    plot_df = plot_df.replace({'topN_replicates': 'Top-N', })
    g = sns.barplot(data=plot_df, x='matching_threshold', hue='method', hue_order=[
        'Intensity\nNon-overlap', 'Top-N', 'Iterative\nExclusion'], y=y, ax=axes[1],
                    palette=palette)
    g.set(ylabel='Annotated features')
    g.set(xlabel='Matching threshold')
    sns.move_legend(g, "upper right", title='Method')
    axes[1].set_title('4 replicates')
    plt.tight_layout()

    return palette
