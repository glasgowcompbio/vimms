import argparse
import configparser
import os
import sys
from abc import ABC, abstractmethod

from vimms.Common import IN_SILICO_OPTIMISE_TOPN, add_log_file, \
    IN_SILICO_OPTIMISE_SMART_ROI, \
    IN_SILICO_OPTIMISE_WEIGHTED_DEW, MSDIAL_DDA_MODE
from vimms.InSilicoSimulation import extract_chemicals, get_timing, \
    extract_timing, run_TopN, run_SmartROI, \
    run_WeightedDEW, extract_boxes, evaluate_boxes_as_dict, \
    evaluate_boxes_as_array, save_counts, string_to_list, \
    plot_counts
from vimms.scripts.msdial_wrapper import run_msdial_batch

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

mode = MSDIAL_DDA_MODE
base_dir = '/Users/joewandy/University of Glasgow/Vinny Davies - CLDS Metabolomics Project/DDAvsDIA/simulation_experiments/DDAvsDIA_rerun'
params_file = os.path.join(base_dir, 'msdial_config', 'Msdial-lcms-dda-Param.txt')
mzml_folder = os.path.join(base_dir, 'results', 'case_v_control_chemicals', 'DDA')
msp_folder = os.path.join(base_dir, 'results', 'chemical_pickles')
msdial_console_app = os.path.join('/Users', 'joewandy', 'MSDIAL ver.4.80 OSX', 'MsdialConsoleApp')
run_msdial_batch(msdial_console_app, mode, params_file, mzml_folder, msp_folder=msp_folder,
                 subdir=True, remove_substring='_cvc')