# Supporting methods needed to run in-silico optimisation of controllers in
# scripts/in_silico_optimise.py
import glob
import json
import os
from pathlib import Path

import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from mass_spec_utils.data_import.mzmine import load_picked_boxes, \
    map_boxes_to_scans
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.mzmine import pick_peaks

from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.Common import set_log_level_warning, set_log_level_debug
from vimms.Controller import TopNController, TopN_SmartRoiController, \
    WeightedDEWController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Roi import RoiBuilderParams


def extract_chemicals(seed_file, params_dict):
    """
    Extract chemicals from a seed file
    :param seed_file: the seed file in mzML format, should be a DDA file
    (containing MS1 and MS2 scans)
    :param params_dict: a dictionary of parameters to extract ROI
    :return: a list of UnknownChemical objects
    """
    logger.info('Seed file = %s' % seed_file)
    logger.info('params = %s' % params_dict)

    rp = RoiBuilderParams(**params_dict)
    cm = ChemicalMixtureFromMZML(seed_file, roi_params=rp)
    dataset = cm.sample(None, 2)
    return dataset


def get_timing(time_dict_str):
    """
    Parses timing information in form of a JSON string to a dictionary
    :param time_dict_str: a string of dictionary in JSON format.
    :return: a dictionary of time information. Key should be the ms-level, 1 or 2, and
    value is the average time of scans at that level
    """
    time_dict = json.loads(time_dict_str)
    time_dict = {int(k): v for k, v in
                 time_dict.items()}  # turn keys from string to int
    logger.debug('Got timing dictionary from config file')
    return time_dict


def extract_timing(seed_file):
    """
    Extracts timing information from a seed file
    :param seed_file: the seed file in mzML format
    If it's a DDA file (containing MS1 and MS2 scans) then both MS1 and MS2
    timing will be extracted.
    If it's only a fullscan file (containing MS1 scans) then only MS1 timing will be extracted.
    :return: a dictionary of time information. Key should be the ms-level, 1 or 2, and
    value is the average time of scans at that level
    """
    logger.debug('Extracting timing dictionary from seed file')
    seed_mzml = MZMLFile(seed_file)

    time_dict = {(1, 1): [], (1, 2): [], (2, 1): [], (2, 2): []}
    for i, s in enumerate(seed_mzml.scans[:-1]):
        current = s.ms_level
        next_ = seed_mzml.scans[i + 1].ms_level
        tup = (current, next_)
        time_dict[tup].append(
            60 * seed_mzml.scans[i + 1].rt_in_minutes - 60 * s.rt_in_minutes)

    is_frag_file = False
    if (1, 2) in time_dict and len(time_dict[(1, 2)]) > 0 and \
            (2, 2) in time_dict and len(time_dict[(2, 2)]) > 0:
        # seed_file must contain timing on (1,2) and (2,2)
        # i.e. it must be a DDA file with MS1 and MS2 scans
        is_frag_file = True

    # construct timing dict in the right format for later use
    new_time_dict = {}
    if is_frag_file:
        # extract ms1 and ms2 timing from fragmentation mzML
        for k, v in time_dict.items():
            if k == (1, 2):
                key = 1
            elif k == (2, 2):
                key = 2
            else:
                continue

            mean = sum(v) / len(v)
            new_time_dict[key] = mean
            logger.debug('%d: %f' % (key, mean))
        assert 1 in new_time_dict and 2 in new_time_dict
    else:
        # extract ms1 timing only from fullscan mzML
        key = 1
        v = time_dict[(1, 1)]
        mean = sum(v) / len(v)
        new_time_dict[key] = mean
        logger.debug('%d: %f' % (key, mean))

    return new_time_dict


def run_TopN(chems, scan_duration, params, out_dir):
    """
    Simulate TopN controller
    :param chems: a list of UnknownChemicals present in the injection
    :param ps: old PeakSampler object, now only used to generate MS2 scans
    (TODO: should be removed as part of issue #46)
    :param params: a dictionary of parameters
    :param out_file: output mzML file
    :param out_dir: output directory
    :return: None
    """
    logger.info('Running TopN simulation')
    logger.info(params)

    out_file = '%s_%s.mzML' % (
        params['controller_name'], params['sample_name'])
    controller = TopNController(params['ionisation_mode'], params['N'],
                                params['isolation_width'], params['mz_tol'],
                                params['rt_tol'], params['min_ms1_intensity'])
    mass_spec = IndependentMassSpectrometer(params['ionisation_mode'], chems,
                                            scan_duration=scan_duration)
    env = Environment(mass_spec, controller, params['min_rt'], params['max_rt'],
                      progress_bar=True, out_dir=out_dir,
                      out_file=out_file)
    logger.info('Generating %s' % out_file)
    env.run()


def run_SmartROI(chems, scan_duration, params, out_dir):
    """
    Simulate SmartROI controller
    :param chems: a list of UnknownChemicals present in the injection
    :param ps: old PeakSampler object, now only used to generate MS2 scans
    (TODO: should be removed as part of issue #46)
    :param params: a dictionary of parameters
    :param out_file: output mzML file
    :param out_dir: output directory
    :return: None
    """
    logger.info('Running SmartROI simulation')
    logger.info(params)
    warn_handler_id = set_log_level_warning()

    iif_values = params['iif_values']
    dp_values = params['dp_values']
    params_list = []
    for iif in iif_values:
        for dp in dp_values:
            # copy params and add additional attributes we need
            copy_params = dict(params)
            copy_params['iif'] = iif
            copy_params['dp'] = dp
            copy_params['chems'] = chems
            copy_params['scan_duration'] = scan_duration
            copy_params['out_dir'] = out_dir
            params_list.append(copy_params)

    # Try to run the controllers in parallel. If fails, then run it serially
    logger.warning('Running controllers in parallel, please wait ...')
    run_serial = False
    try:
        rc = ipp.Client()
        dview = rc[:]  # use all engines
        with dview.sync_imports():
            pass
        dview.map_sync(run_single_SmartROI, params_list)
    except OSError:  # cluster has not been started
        run_serial = True
    except ipp.error.TimeoutError:  # takes too long to run
        run_serial = True

    if run_serial:  # if any exception from above, try to run it serially
        logger.warning(
            'IPython cluster not found, running controllers in serial mode')
        for copy_params in params_list:
            run_single_SmartROI(copy_params)

    set_log_level_debug(remove_id=warn_handler_id)


def run_single_SmartROI(params):
    out_file = 'SMART_{}_{}_{}.mzml'.format(params['sample_name'],
                                            params['iif'], params['dp'])
    logger.warning('Generating %s' % out_file)
    if os.path.isfile(os.path.join(params['out_dir'], out_file)):
        logger.warning('Already done')
        return

    intensity_increase_factor = params[
        'iif']  # fragment ROI again if intensity increases 10 fold
    drop_perc = params['dp'] / 100
    reset_length_seconds = 1e6  # set so reset never happens

    controller = TopN_SmartRoiController(params['ionisation_mode'],
                                         params['isolation_width'],
                                         params['mz_tol'],
                                         params['min_ms1_intensity'],
                                         params['min_roi_intensity'],
                                         params['min_roi_length'],
                                         N=params['N'], rt_tol=params['rt_tol'],
                                         min_roi_length_for_fragmentation=params[
                                             'min_roi_length_for_fragmentation'],
                                         reset_length_seconds=reset_length_seconds,
                                         intensity_increase_factor=intensity_increase_factor,
                                         drop_perc=drop_perc)

    mass_spec = IndependentMassSpectrometer(params['ionisation_mode'],
                                            params['chems'],
                                            scan_duration=params[
                                                'scan_duration'])
    env = Environment(mass_spec, controller, params['min_rt'], params['max_rt'],
                      progress_bar=True,
                      out_dir=params['out_dir'], out_file=out_file)
    env.run()


def run_WeightedDEW(chems, scan_duration, params, out_dir):
    """
    Simulate WeightedDEW controller
    :param chems: a list of UnknownChemicals present in the injection
    :param ps: old PeakSampler object, now only used to generate MS2 scans
    (TODO: should be removed as part of issue #46)
    :param params: a dictionary of parameters
    :param out_file: output mzML file
    :param out_dir: output directory
    :return: None
    """
    logger.info('Running WeightedDEW simulation')
    logger.info(params)
    warn_handler_id = set_log_level_warning()

    t0_values = params['t0_values']
    rt_tol_values = params['rt_tol_values']
    params_list = []
    for t0 in t0_values:
        for r in rt_tol_values:
            # copy params and add additional attributes we need
            copy_params = dict(params)
            copy_params['t0'] = t0
            copy_params['r'] = r
            copy_params['chems'] = chems
            copy_params['scan_duration'] = scan_duration
            copy_params['out_dir'] = out_dir
            params_list.append(copy_params)

    # Try to run the controllers in parallel. If fails, then run it serially
    logger.warning('Running controllers in parallel, please wait ...')
    try:
        import ipyparallel as ipp
        rc = ipp.Client()
        dview = rc[:]  # use all engines
        with dview.sync_imports():
            pass
        dview.map_sync(run_single_WeightedDEW, params_list)
    except OSError:  # cluster has not been started
        run_serial = True
    except ipp.error.TimeoutError:  # takes too long to run
        run_serial = True

    if run_serial:  # if any exception from above, try to run it serially
        logger.warning(
            'IPython cluster not found, running controllers in serial mode')
        for copy_params in params_list:
            run_single_WeightedDEW(copy_params)

    set_log_level_debug(remove_id=warn_handler_id)


def run_single_WeightedDEW(params):
    out_file = 'WeightedDEW_{}_{}_{}.mzml'.format(params['sample_name'],
                                                  params['t0'], params['r'])
    logger.warning('Generating %s' % out_file)
    if os.path.isfile(os.path.join(params['out_dir'], out_file)):
        logger.warning('Already done')
        return
    if params['t0'] > params['r']:
        logger.warning('Impossible combination')
        return

    controller = WeightedDEWController(params['ionisation_mode'], params['N'],
                                       params['isolation_width'],
                                       params['mz_tol'], params['r'],
                                       params['min_ms1_intensity'],
                                       exclusion_t_0=params['t0'],
                                       log_intensity=True)
    mass_spec = IndependentMassSpectrometer(params['ionisation_mode'],
                                            params['chems'],
                                            scan_duration=params[
                                                'scan_duration'])
    env = Environment(mass_spec, controller, params['min_rt'], params['max_rt'],
                      progress_bar=True,
                      out_dir=params['out_dir'], out_file=out_file)
    env.run()


def string_to_list(my_str, convert=None):
    """
    Convert a string representation of list into list
    :param my_str: a string representation of list, e.g. '[1, 2, 3]'
    :param convert: a function to convert data type of each element, e.g. float
    :return: a list, e.g. [1, 2, 3]
    """
    if len(my_str) == 0:
        return []
    my_list = my_str.strip('][').split(', ')

    if convert is not None:
        my_list = list(map(convert, my_list))
    return my_list


def extract_boxes(seed_file, out_dir, mzmine_command, xml_file):
    """
    Extract peak picked boxes using MzMine2 peak picking
    :param seed_file: the seed file in mzML format, should be a DDA file
    (containing MS1 and MS2 scans)
    :param mzmine_command: path to MzMine2 batch file
    :param xml_file: path to MzMine2 XML config file
    :return: a list of boxes
    """
    # construct the path to the resulting peak picked CSV
    seed_picked_peaks_csv = get_peak_picked_csv(seed_file)
    logger.info('Peak picking, results will be in %s' % seed_picked_peaks_csv)

    # run peak picking using MzMine2
    pick_peaks([seed_file], xml_template=xml_file, output_dir=out_dir,
               mzmine_command=mzmine_command)

    # the peak picked csv must exist at this point
    assert Path(seed_picked_peaks_csv).is_file()
    boxes = load_picked_boxes(seed_picked_peaks_csv)
    logger.info('Loaded %d boxes from the seed file' % len(boxes))
    return boxes


def get_peak_picked_csv(seed_file):
    """
    From the seed file returns the path to the peak picked csv file from mzmine
    :param seed_file: path to the seed file
    :return: path to the peak picked csv file from mzine
    """
    base_name = os.path.basename(seed_file)
    seed_picked_peaks = os.path.splitext(base_name)[0] + '_box.csv'
    seed_dir = os.path.split(seed_file)[0]
    seed_picked_peaks_csv = os.path.join(seed_dir, seed_picked_peaks)
    return seed_picked_peaks_csv


def evaluate_boxes_as_dict(boxes, out_dir):
    counts = {}
    for filename in glob.glob(os.path.join(out_dir, '*.mzML')):
        basename = os.path.basename(filename)
        mzml = MZMLFile(filename)
        scans2boxes, boxes2scans = map_boxes_to_scans(mzml, boxes,
                                                      half_isolation_window=0)
        c = len(boxes2scans)
        logger.info('- %s: found %d boxes with scans' % (basename, c))
        counts[basename] = c
    logger.debug(counts)
    return counts


def evaluate_boxes_as_array(boxes, out_dir, yticks, xticks, pattern, params):
    sample_name = params['sample_name']
    counts = np.zeros((len(yticks), len(xticks)))
    for i, y in enumerate(yticks):
        for j, x in enumerate(xticks):
            try:
                fname = pattern.format(sample_name, y, x)
                mz_file = MZMLFile(os.path.join(out_dir, fname))
                scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes,
                                                              half_isolation_window=0)
                counts[i, j] = len(boxes2scans)
            except FileNotFoundError:
                counts[i, j] = np.nan
            logger.debug(counts)
    return counts


def save_counts(counts, out_dir, controller_name, sample_name):
    fname = '%s_%s_counts.csv' % (controller_name, sample_name)
    out_csv = os.path.join(out_dir, fname)
    np.savetxt(out_csv, counts, delimiter=",")


def plot_counts(counts, out_file, title, xlabel, xticks, ylabel, yticks):
    plt.rcParams.update({'font.size': 16})
    plt.imshow(counts, aspect='auto')
    plt.yticks(range(len(yticks)), yticks)
    plt.xticks(range(len(xticks)), xticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
