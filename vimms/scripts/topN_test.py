import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import os
import argparse

import numpy as np

import pymzml
from loguru import logger

from vimms.Roi import RoiBuilderParams
from vimms.Chemicals import ChemicalMixtureFromMZML
from vimms.ChemicalSamplers import MzMLScanTimeSampler

from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController, AdvancedParams
from vimms.Environment import Environment
from vimms.Common import POSITIVE, load_obj, save_obj, create_if_not_exist, \
    set_log_level_warning, set_log_level_debug


def parse_args():
    parser = argparse.ArgumentParser(description='VIMMS simulation.')
    parser.add_argument('--at_least_one_point_above', type=float, default=1E5,
                        help='The minimum intensity value for ROI extraction.')
    parser.add_argument('--min_rt', type=int, default=0,
                        help='The minimum retention time for Top-N.')
    parser.add_argument('--max_rt', type=int, default=7700,
                        help='The maximum retention time for Top-N.')
    parser.add_argument('--isolation_window', type=int, default=1,
                        help='The isolation window for Top-N.')
    parser.add_argument('--N', type=int, default=15,
                        help='The Top N value.')
    parser.add_argument('--rt_tol', type=int, default=30,
                        help='The retention time tolerance for Top-N.')
    parser.add_argument('--mz_tol', type=int, default=10,
                        help='The mass to charge ratio tolerance for Top-N.')
    parser.add_argument('--min_ms1_intensity', type=int, default=5000,
                        help='The minimum MS1 intensity for Top-N.')
    parser.add_argument('--default_ms1_scan_window_start', type=float, default=310.0,
                        help='The start of the default MS1 scan window.')
    parser.add_argument('--default_ms1_scan_window_end', type=float, default=2000.0,
                        help='The end of the default MS1 scan window.')
    parser.add_argument('--exclude_after_n_times', type=int, default=2,
                        help='The number of times to exclude after in DEW parameters.')
    parser.add_argument('--exclude_t0', type=int, default=15,
                        help='The exclude t0 value in DEW parameters.')
    parser.add_argument('--deisotope', type=bool, default=True,
                        help='Whether to perform deisotoping or not.')
    parser.add_argument('--charge_range_start', type=int, default=2,
                        help='The start of the charge range for filtering.')
    parser.add_argument('--charge_range_end', type=int, default=6,
                        help='The end of the charge range for filtering.')
    parser.add_argument('--out_dir', type=str, default='topN_test',
                        help='The directory where the output files will be stored.')
    parser.add_argument('--pbar', type=bool, default=True,
                        help='If true, progress bar will be shown.')
    parser.add_argument('--in_mzml', type=str, default='BSA_100fmol__recon_1ul_1.mzML',
                        help='The filename of the input mzML file.')
    parser.add_argument('--out_mzml', type=str, default='output.mzML',
                        help='The filename of the output mzML file.')
    args = parser.parse_args()
    return args


def get_input_filenames(at_least_one_point_above, base_dir):
    formatted_number = '%.0e' % at_least_one_point_above
    formatted_number = formatted_number.replace('e', 'E').replace('+', '')
    chem_file = os.path.join(base_dir, f'chems_{formatted_number}.p')
    st_file = os.path.join(base_dir, f'scan_timing_{formatted_number}.p')
    return chem_file, st_file


def extract_scan_timing(mzml_file, st_file):
    if os.path.isfile(st_file):
        st = load_obj(st_file)
    else:
        # extract timing from mzML file by taking the mean of MS1 scan durations
        logger.debug(f'Extracting scan timing from {mzml_file}')
        st = MzMLScanTimeSampler(mzml_file, use_mean=True, use_ms1_count=True)
        save_obj(st, st_file)
    return st


def extract_chems(mzml_file, chem_file, at_least_one_point_above):
    if os.path.isfile(chem_file):
        dataset = load_obj(chem_file)
    else:
        logger.debug(f'Extracting chems from {mzml_file}')
        rp = RoiBuilderParams(at_least_one_point_above=at_least_one_point_above)
        cm = ChemicalMixtureFromMZML(mzml_file, roi_params=rp)
        dataset = cm.sample(None, 2)
        logger.debug(f'Extracted {len(dataset)} chems')
        save_obj(dataset, chem_file)
    return dataset


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


def main(args):
    # check input and output paths
    assert os.path.isfile(args.in_mzml), 'Input mzML file %s is not found!' % args.in_mzml
    out_dir = os.path.abspath(args.out_dir)
    create_if_not_exist(out_dir)

    # Format output file names
    chem_file, st_file = get_input_filenames(args.at_least_one_point_above, out_dir)

    # extract chems and scan timing from mzml file
    dataset = extract_chems(args.in_mzml, chem_file, args.at_least_one_point_above)
    st = extract_scan_timing(args.in_mzml, st_file)

    # simulate Top-N
    run_simulation(args, dataset, st, out_dir)


def run_simulation(args, dataset, st, out_dir):

    # Top-N parameters
    rt_range = [(args.min_rt, args.max_rt)]
    min_rt = rt_range[0][0]
    max_rt = rt_range[0][1]
    isolation_window = args.isolation_window
    N = args.N
    rt_tol = args.rt_tol
    mz_tol = args.mz_tol
    min_ms1_intensity = args.min_ms1_intensity
    default_ms1_scan_window = (
        args.default_ms1_scan_window_start, args.default_ms1_scan_window_end)

    # DEW, isotope and charge filtering parameters
    exclude_after_n_times = args.exclude_after_n_times
    exclude_t0 = args.exclude_t0
    deisotope = args.deisotope
    charge_range = (args.charge_range_start, args.charge_range_end)

    # create controller and mass spec objects
    params = AdvancedParams(default_ms1_scan_window=default_ms1_scan_window)
    mass_spec = IndependentMassSpectrometer(POSITIVE, dataset, scan_duration=st)
    controller = TopNController(
        POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity,
        advanced_params=params, exclude_after_n_times=exclude_after_n_times,
        exclude_t0=exclude_t0, deisotope=deisotope, charge_range=charge_range)

    # create an environment to run both the mass spec and controller
    env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=args.pbar)

    # set the log level to WARNING so we don't see too many messages when environment is running
    set_log_level_warning()

    # run the simulation
    env.run()
    set_log_level_debug()
    env.write_mzML(out_dir, args.out_mzml)


if __name__ == '__main__':
    args = parse_args()
    main(args)