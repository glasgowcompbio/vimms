import glob
import os

import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.mzmine import pick_peaks

from vimms.Agent import TopNDEWAgent
from vimms.Box import BoxGrid
from vimms.BoxManager import BoxManager, BoxSplitter
from vimms.BoxVisualise import PlotPoints
from vimms.Common import load_obj
from vimms.Controller import AgentBasedController, TopN_SmartRoiController, TopNController, \
    WeightedDEWController, AIF, SWATH, AdvancedParams
from vimms.Controller.box import IntensityNonOverlapController, NonOverlapController
from vimms.Environment import Environment
from vimms.Evaluation import evaluate_multi_peak_roi_aligner
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiAligner


def run_experiment(result_folder, sample_list, controller_names, experiment_params):
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

    # Try to run the controllers in parallel. If fails, then run it serially
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
    except ipp.error.TimeoutError:  # takes too long to run
        run_serial = True

    if run_serial:  # if any exception from above, try to run it serially
        logger.warning(
            'IPython cluster not found, running controllers in serial mode')
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


def plot_results(controller_names, eval_res, suptitle=None, outfile=None):
    sns.set_context('poster')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for exp_name in controller_names:
        results = eval_res[exp_name]
        try:
            coverages = results["cumulative_coverage_prop"]
        except KeyError:
            coverages = results['cumulative_coverage_proportion']

        try:
            intensity_proportions = results["cumulative_coverage_intensities_prop"]
        except KeyError:
            intensity_proportions = results["cumulative_intensity_proportion"]

        xis = list(range(1, len(coverages) + 1))

        ax1.set(xlabel="Num. Runs", ylabel="Cumulative Coverage Proportion",
                title="Multi-Sample Cumulative Coverage")
        ax1.plot(xis, coverages, label=exp_name)
        ax1.legend(loc='lower right', bbox_to_anchor=(1, 0.05))

        ax2.set(xlabel="Num. Runs", ylabel="Cumulative Intensity Proportion",
                title="Multi-Sample Cumulative Intensity Proportion")
        ax2.plot(xis, intensity_proportions, label=exp_name)
        ax2.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
    fig.set_size_inches(18.5, 10.5)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=32)
    if outfile is not None:
        plt.savefig(outfile, facecolor='white', transparent=False)


def print_results(controller_names, eval_res):
    for exp_name in controller_names:
        print(exp_name)
        print(f"Cumulative Coverage: {successes(eval_res[exp_name])}")
        print(f"Cumulative Coverage Proportion: {eval_res[exp_name]['cumulative_coverage_prop']}")
        print(
            f"Cumulative Intensity Proportion: {eval_res[exp_name]['cumulative_coverage_intensities_prop']}")
        print()


def boxplot_results(df, suptitle=None, outfile=None):
    sns.set_context(context='poster', font_scale=1, rc=None)

    figsize = (20, 20)
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    sns.boxplot(x='sample_num', y='coverage_prop', hue='controller', data=df, ax=axes[0])
    axes[0].set_title('Multi-sample Cumulative Coverage')

    sns.boxplot(x='sample_num', y='intensity_prop', hue='controller', data=df, ax=axes[1])
    axes[1].set_title('Multi-sample Cumulative Intensity Proportion')

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=32)
    plt.tight_layout()

    if outfile is not None:
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
