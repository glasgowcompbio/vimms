import gc

import pandas as pd
from loguru import logger

from vimms.Agent import TopNDEWAgent
from vimms.Box import BoxGrid
from vimms.BoxManager import BoxManager, BoxSplitter
from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.Common import CONTROLLER_FULLSCAN, CONTROLLER_TOPN, CONTROLLER_TOPN_EXCLUSION, \
    CONTROLLER_SWATH, CONTROLLER_AIF, CONTROLLER_NON_OVERLAP, CONTROLLER_INTENSITY_NON_OVERLAP, \
    CONTROLLER_INTENSITY_ROI_EXCLUSION, CONTROLLER_HARD_ROI_EXCLUSION
from vimms.Controller import SimpleMs1Controller
from vimms.Controller import TopNController, AIF, SWATH, AgentBasedController
from vimms.Controller.box import NonOverlapController, IntensityNonOverlapController, \
    IntensityRoIExcludeController, HardRoIExcludeController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiBuilderParams


def extract_chemicals(seed_file, ionisation_mode):
    rp = RoiBuilderParams(min_roi_length=3, at_least_one_point_above=1000)
    cm = ChemicalMixtureFromMZML(seed_file, roi_params=rp)
    dataset = cm.sample(None, 2, source_polarity=ionisation_mode)
    return dataset


def run_batch(initial_runs, controller_repeat, experiment_params, samples,
              pbar, max_time, ionisation_mode, use_instrument, use_column,
              ref_dir, dataset, out_dir):
    scan_duration_dict = experiment_params['scan_duration_dict']

    # perform initial blank and QC runs here
    for sample in initial_runs:
        controller = select_controller(CONTROLLER_FULLSCAN, experiment_params, None, None)
        out_file = get_out_file(CONTROLLER_FULLSCAN, sample, 0)
        run_controller(use_instrument, ref_dir, dataset, scan_duration_dict,
                       pbar, max_time, ionisation_mode, use_column, controller, out_dir, out_file)

    # loop through each controller
    for controller_name in controller_repeat:

        # sharable across all samples all replicates
        agent = None
        grid = None
        if controller_name == CONTROLLER_TOPN_EXCLUSION:
            agent = make_agent(experiment_params)
        elif controller_name in [CONTROLLER_NON_OVERLAP, CONTROLLER_HARD_ROI_EXCLUSION]:
            grid = make_grid(experiment_params, False)
        elif controller_name in [CONTROLLER_INTENSITY_NON_OVERLAP,
                                 CONTROLLER_INTENSITY_ROI_EXCLUSION]:
            grid = make_grid(experiment_params, True)

        # find how many replicates have been specified
        sample_to_process, repeat = controller_repeat[controller_name]
        for i in range(repeat):

            # for each sample in the replicate, check if we should process it
            for sample in samples:
                if sample not in sample_to_process:
                    continue

                # if yes, run the controller
                controller = select_controller(controller_name, experiment_params, agent, grid)
                out_file = get_out_file(controller_name, sample, i)
                run_controller(use_instrument, ref_dir, dataset, scan_duration_dict,
                               pbar, max_time, ionisation_mode, use_column, controller, out_dir,
                               out_file)
                # fname = os.path.join(out_dir, out_file+'.controller')
                # save_obj(controller, fname)
                del controller
                gc.collect()


# a variant of run_batch but for exhaustive fragmentation (experiment 3)
def run_batch_exhaustive(initial_runs, controller_repeat, experiment_params, samples,
                         pbar, max_time, ionisation_mode, use_instrument, use_column,
                         ref_dir, dataset, out_dir):
    scan_duration_dict = experiment_params['scan_duration_dict']

    # perform initial blank and QC runs here
    for sample in initial_runs:
        controller = select_controller(CONTROLLER_FULLSCAN, experiment_params, None, None)
        out_file = get_out_file(CONTROLLER_FULLSCAN, sample, 0)
        run_controller(use_instrument, ref_dir, dataset, scan_duration_dict,
                       pbar, max_time, ionisation_mode, use_column, controller, out_dir, out_file)

    # loop through each controller
    for controller_name in controller_repeat:

        # find how many replicates have been specified
        sample_to_process, repeat = controller_repeat[controller_name]

        # for each sample in the replicate, check if we should process it
        for sample in samples:
            if sample not in sample_to_process:
                continue

            # reset sharable after each sample
            agent = None
            grid = None
            if controller_name == CONTROLLER_TOPN_EXCLUSION:
                agent = make_agent(experiment_params)
            elif controller_name in [CONTROLLER_NON_OVERLAP, CONTROLLER_HARD_ROI_EXCLUSION]:
                grid = make_grid(experiment_params, False)
            elif controller_name in [CONTROLLER_INTENSITY_NON_OVERLAP,
                                     CONTROLLER_INTENSITY_ROI_EXCLUSION]:
                grid = make_grid(experiment_params, True)

            for i in range(repeat):
                # if yes, run the controller
                controller = select_controller(controller_name, experiment_params, agent, grid)
                out_file = get_out_file(controller_name, sample, i)
                run_controller(use_instrument, ref_dir, dataset, scan_duration_dict,
                               pbar, max_time, ionisation_mode, use_column, controller, out_dir,
                               out_file)
                # fname = os.path.join(out_dir, out_file+'.controller')
                # save_obj(controller, fname)
                del controller
                gc.collect()


def make_grid(experiment_params, split_grid):
    logger.warning('Grid initialised, split_grid=%s' % split_grid)
    grid_params = experiment_params['grid_params']
    min_measure_rt = grid_params['min_measure_rt']
    max_measure_rt = grid_params['max_measure_rt']
    rt_box_size = grid_params['rt_box_size']
    mz_box_size = grid_params['mz_box_size']
    grid = BoxManager(
        box_geometry=BoxGrid(min_measure_rt, max_measure_rt, rt_box_size, 0, 1500, mz_box_size),
        box_splitter=BoxSplitter(split=split_grid)
    )
    return grid


def make_agent(experiment_params):
    logger.warning('TopNDEWAgent initialised')
    topN_params = experiment_params['topN_params']
    agent = TopNDEWAgent(**topN_params)
    return agent


def get_out_file(controller_name, sample, i):
    out_file = "{}_{}_{}.mzML".format(controller_name, sample, i)
    return out_file


def select_controller(controller_name, experiment_params, agent, grid):
    if controller_name == CONTROLLER_FULLSCAN:
        controller = SimpleMs1Controller()

    elif controller_name == CONTROLLER_TOPN:
        topN_params = experiment_params['topN_params']
        controller = TopNController(**topN_params)

    elif controller_name == CONTROLLER_TOPN_EXCLUSION:
        controller = AgentBasedController(agent)

    elif controller_name == CONTROLLER_SWATH:
        SWATH_params = experiment_params['SWATH_params']
        min_mz = SWATH_params['min_mz']
        max_mz = SWATH_params['max_mz']
        width = SWATH_params['width']
        scan_overlap = SWATH_params['scan_overlap']
        controller = SWATH(min_mz, max_mz, width, scan_overlap=scan_overlap)

    elif controller_name == CONTROLLER_AIF:
        AIF_params = experiment_params['AIF_params']
        ms1_source_cid_energy = AIF_params['ms1_source_cid_energy']
        controller = AIF(ms1_source_cid_energy)

    elif controller_name == CONTROLLER_NON_OVERLAP:
        non_overlap_params = get_non_overlap_params(experiment_params)
        controller = NonOverlapController(grid=grid, **non_overlap_params)

    elif controller_name == CONTROLLER_INTENSITY_NON_OVERLAP:
        non_overlap_params = get_non_overlap_params(experiment_params)
        controller = IntensityNonOverlapController(grid=grid, **non_overlap_params)

    elif controller_name == CONTROLLER_INTENSITY_ROI_EXCLUSION:
        non_overlap_params = get_non_overlap_params(experiment_params)
        controller = IntensityRoIExcludeController(grid=grid, **non_overlap_params)

    elif controller_name == CONTROLLER_HARD_ROI_EXCLUSION:
        non_overlap_params = get_non_overlap_params(experiment_params)
        controller = HardRoIExcludeController(grid=grid, **non_overlap_params)

    else:
        logger.warning('Unknown controller: %s' % controller_name)
        controller = None

    return controller


def get_non_overlap_params(experiment_params):
    topN_params = experiment_params['topN_params']
    non_overlap_params = {**topN_params, **experiment_params['non_overlap_params']}
    non_overlap_scoring = experiment_params['non_overlap_scoring']

    # check whether to use smartroi exclusion
    if non_overlap_scoring['use_smartroi_exclusion']:
        smartroi_params = experiment_params['smartroi_params']
        non_overlap_params = {**non_overlap_params, **smartroi_params}

    # check whether to use weighteddew exclusion
    elif non_overlap_scoring['use_weighteddew_exclusion']:
        weighteddew_params = experiment_params['weighteddew_params']
        non_overlap_params = {**non_overlap_params, **weighteddew_params}

    return non_overlap_params


def run_controller(use_instrument, ref_dir, dataset, scan_duration_dict,
                   pbar, max_time, ionisation_mode, use_column, controller, out_dir, out_file):
    logger.warning(out_file)
    if use_instrument:
        from vimms_fusion.MassSpec import IAPIMassSpectrometer
        from vimms_fusion.Environment import IAPIEnvironment

        mass_spec = IAPIMassSpectrometer(ionisation_mode, ref_dir, filename=None,
                                         show_console_logs=False,
                                         use_column=use_column)
        with IAPIEnvironment(mass_spec, controller, max_time, progress_bar=pbar, out_dir=out_dir,
                              out_file=out_file) as env:
            env.run()
        del mass_spec, env
    else:
        mass_spec = IndependentMassSpectrometer(ionisation_mode, dataset,
                                                scan_duration=scan_duration_dict)
        env = Environment(mass_spec, controller, 0, max_time, progress_bar=pbar, out_dir=out_dir,
                          out_file=out_file)
        env.run()


def generate_sequence_df(initial_runs, controller_repeat, samples, position, raw_output_path,
                         blank_method_path, instrument_method_path, exhaustive=False):
    all_runs = []

    for sample in initial_runs:
        method_file, sample_type = select_sample_type_and_method_file(
            blank_method_path, instrument_method_path, sample)
        sample_position = position[sample]
        controller_name = CONTROLLER_FULLSCAN
        out_file = "{}_{}".format(controller_name, sample)
        row = generate_row(method_file, out_file, raw_output_path, sample_position, sample_type)
        all_runs.append(row)

    if not exhaustive:  # normal setup

        for controller_name in controller_repeat:
            sample_to_process, repeat = controller_repeat[controller_name]
            for i in range(repeat):

                for sample in samples:
                    if sample not in sample_to_process:
                        continue

                    method_file, sample_type = select_sample_type_and_method_file(
                        blank_method_path, instrument_method_path, sample)
                    sample_position = position[sample]
                    out_file = "{}_{}_{}".format(controller_name, sample, i)
                    row = generate_row(method_file, out_file, raw_output_path, sample_position,
                                       sample_type)
                    all_runs.append(row)

    else:  # exhaustive setup

        for controller_name in controller_repeat:
            sample_to_process, repeat = controller_repeat[controller_name]
            for sample in samples:
                if sample not in sample_to_process:
                    continue

                for i in range(repeat):
                    method_file, sample_type = select_sample_type_and_method_file(
                        blank_method_path, instrument_method_path, sample)
                    sample_position = position[sample]
                    out_file = "{}_{}_{}".format(controller_name, sample, i)
                    row = generate_row(method_file, out_file, raw_output_path, sample_position,
                                       sample_type)
                    all_runs.append(row)

    headers = ['Sample Type', 'File Name', 'Sample ID', 'Path', 'Instrument Method',
               'Process Method', 'Calibration File', 'Position', 'Inj Vol',
               'Level', 'Sample Wt', 'Sample Vol', 'ISTD Amt', 'Dil Factor',
               'L1 Study', 'L2 Client', 'L3 Laboratory', 'L4 Company', 'L5 Phone', 'Comment',
               'Sample Name']
    df = pd.DataFrame(all_runs, columns=headers)
    return df


def select_sample_type_and_method_file(blank_method_path, instrument_method_path, sample):
    if 'cmw' in sample.lower():
        sample_type = 'Unknown'
        method_file = instrument_method_path
    elif 'blank' in sample.lower():
        sample_type = 'Blank'
        method_file = blank_method_path
    else:
        sample_type = 'Unknown'
        method_file = instrument_method_path
    return method_file, sample_type


def generate_row(method_file, out_file, raw_output_path, sample_position, sample_type):
    sample_id = 1
    process_method = None
    calibration_file = None
    injection_vol = 10
    level = None
    sample_wt = 0
    sample_vol = 0
    istd_amt = 0
    dil_factor = 1
    l1_study = None
    l2_client = None
    l3_laboratory = None
    l4_company = None
    l5_phone = None
    comment = None
    sample_name = None
    row = [sample_type, out_file, sample_id, raw_output_path, method_file,
           process_method, calibration_file, sample_position, injection_vol,
           level, sample_wt, sample_vol, istd_amt, dil_factor,
           l1_study, l2_client, l3_laboratory, l4_company, l5_phone, comment, sample_name]
    return row


def write_sequence_csv(df, out_file):
    with open(out_file, 'w') as f:
        df.to_csv(f, header=True, index=False)
    line_prepender(out_file, 'Bracket Type=4,')


def line_prepender(filename, line):
    # https://stackoverflow.com/questions/5914627/prepend-line-to-beginning-of-a-file
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
