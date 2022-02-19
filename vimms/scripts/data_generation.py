# data generation script
from vimms.Utils import write_msp
from vimms.Environment import Environment
from vimms.Noise import UniformSpikeNoise
from vimms.Controller import TopNController, SWATH
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.ChemicalSamplers import DatabaseFormulaSampler, \
    UniformRTAndIntensitySampler, UniformMS2Sampler
from vimms.Chemicals import ChemicalMixtureCreator
import argparse
import sys

import numpy as np
from loguru import logger

from vimms.Common import DEFAULT_MS1_SCAN_WINDOW, load_obj, \
    ADDUCT_DICT_POS_MH, set_log_level_warning

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


DEFAULT_RT_RANGE = (100, 500)
POSITIVE_IONISATION_MODE = "positive"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Limited dataset creation')
    parser.add_argument('output_mzml_file', type=str)
    parser.add_argument('--min_mz', dest='min_mz',
                        default=DEFAULT_MS1_SCAN_WINDOW[0], type=float)
    parser.add_argument('--max_mz', dest='max_mz',
                        default=DEFAULT_MS1_SCAN_WINDOW[1], type=float)
    parser.add_argument('--min_rt', dest='min_rt', default=DEFAULT_RT_RANGE[0],
                        type=float)
    parser.add_argument('--max_rt', dest='max_rt', default=DEFAULT_RT_RANGE[1],
                        type=float)
    parser.add_argument('--min_ms1_sampling_intensity',
                        dest='min_ms1_sampling_intensity', default=1e3,
                        type=float)
    parser.add_argument('--max_ms1_sampling_intensity',
                        dest='max_ms1_sampling_intensity', default=1e9,
                        type=float)
    parser.add_argument('--formula_database_file',
                        dest='formula_database_file',
                        type=str)
    parser.add_argument('--n_chems', dest='n_chems', default=10, type=int)
    parser.add_argument('--ms_levels', dest='ms_levels', default=2, type=int)
    parser.add_argument('--output_msp_file', dest='output_msp_file',
                        default=None, type=str)
    parser.add_argument('--spike_max', dest='spike_max', default=1000,
                        type=float)
    parser.add_argument('--output_swath_file', dest='output_swath_file',
                        type=str, default=None)
    parser.add_argument('--print_chems', dest='print_chems',
                        action='store_true')

    args = parser.parse_args()

    formula_database = load_obj(args.formula_database_file)

    logger.debug("Loaded {} formulas".format(len(formula_database)))

    fs = DatabaseFormulaSampler(formula_database, min_mz=args.min_mz,
                                max_mz=args.max_mz)

    ri = UniformRTAndIntensitySampler(min_rt=args.min_rt, max_rt=args.max_rt,
                                      min_log_intensity=np.log(
                                          args.min_ms1_sampling_intensity),
                                      max_log_intensity=np.log(
                                          args.max_ms1_sampling_intensity))
    cs = UniformMS2Sampler()

    cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri,
                                ms2_sampler=cs,
                                adduct_prior_dict=ADDUCT_DICT_POS_MH)

    dataset = cm.sample(args.n_chems, args.ms_levels)

    if args.print_chems:
        logger.debug("Sampled chems")
        for chem in dataset:
            logger.debug(chem)

    if args.output_msp_file is not None:
        write_msp(dataset, args.output_msp_file)

    spike_noise = UniformSpikeNoise(0.01, args.spike_max)

    ms = IndependentMassSpectrometer(POSITIVE_IONISATION_MODE, dataset,
                                     spike_noise=spike_noise)

    controller = TopNController(POSITIVE_IONISATION_MODE, 10, 0.7, 0.01, 15,
                                1e3)

    env = Environment(ms, controller, min_time=args.min_rt - 50,
                      max_time=args.max_rt + 50)

    set_log_level_warning()
    env.run()

    env.write_mzML(None, args.output_mzml_file)

    if args.output_swath_file is not None:
        sw = SWATH(args.min_mz, args.max_mz, 100, 0.0)
        ms = IndependentMassSpectrometer(POSITIVE_IONISATION_MODE, dataset,
                                         spike_noise=spike_noise)
        env = Environment(ms, sw, min_time=args.min_rt - 50,
                          max_time=args.max_rt + 50)
        env.run()
        env.write_mzML(None, args.output_swath_file)
