import argparse
import configparser
import os
import sys
from abc import ABC, abstractmethod

from vimms.Common import IN_SILICO_OPTIMISE_TOPN, add_log_file, \
    IN_SILICO_OPTIMISE_SMART_ROI, \
    IN_SILICO_OPTIMISE_WEIGHTED_DEW
from vimms.InSilicoSimulation import extract_chemicals, get_timing, \
    extract_timing, run_TopN, run_SmartROI, \
    run_WeightedDEW, extract_boxes, evaluate_boxes_as_dict, \
    evaluate_boxes_as_array, save_counts, string_to_list, \
    plot_counts

sys.path.append('..')
sys.path.append('../..')  # if running in this folder


class InSilicoSimulator(ABC):
    def __init__(self, sample_name, seed_file, out_dir, controller_name,
                 config_parser):
        self.sample_name = sample_name
        self.seed_file = seed_file
        self.out_dir = out_dir
        self.controller_name = controller_name
        self.config_parser = config_parser

    def run(self):
        # get the chemicals, timing, peak sample object and parameters
        chems = self.get_chems()
        scan_duration = self.get_scan_duration()
        params = self.get_params()

        # simulate controller and evaluate performance
        self.simulate(chems, scan_duration, params)
        self.evaluate(params)

    def get_chems(self):
        # extract chemicals from seed_file
        params_dict = {
            'mz_tol': self.config_parser.getint('roi_extraction', 'mz_tol'),
            'mz_units': self.config_parser.get('roi_extraction', 'mz_units'),
            'min_length': self.config_parser.getint('roi_extraction',
                                                    'min_length'),
            'min_intensity': self.config_parser.getint('roi_extraction',
                                                       'min_intensity'),
            'start_rt': self.config_parser.getint('experiment', 'min_rt'),
            'stop_rt': self.config_parser.getint('experiment', 'max_rt')
        }
        chems = extract_chemicals(self.seed_file, params_dict)
        return chems

    def get_scan_duration(self):
        # if provided, read timing information from config
        # otherwise extract timing from the seed file too
        # parse time dict, this really should be computed from the data
        time_dict_str = self.config_parser.get('simulation', 'scan_duration')
        time_dict = get_timing(time_dict_str) if len(
            time_dict_str) > 0 else extract_timing(self.seed_file)
        return time_dict

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def evaluate(self, params):
        pass


class TopNSimulator(InSilicoSimulator):
    def get_params(self):
        # get experiment parameters
        ionisation_mode = self.config_parser.get('experiment',
                                                 'ionisation_mode')
        isolation_width = self.config_parser.getfloat('experiment',
                                                      'isolation_width')
        min_rt = self.config_parser.getfloat('experiment', 'min_rt')
        max_rt = self.config_parser.getint('experiment', 'max_rt')

        # get simulation parameters
        N = self.config_parser.getint('simulation', 'N')
        mz_tol = self.config_parser.getint('simulation', 'mz_tol')
        rt_tol = self.config_parser.getint('simulation', 'rt_tol')
        min_ms1_intensity = self.config_parser.getint('simulation',
                                                      'min_ms1_intensity')

        params = {
            'controller_name': self.controller_name,
            'ionisation_mode': ionisation_mode,
            'sample_name': self.sample_name,
            'isolation_width': isolation_width,
            'N': N,
            'mz_tol': mz_tol,
            'rt_tol': rt_tol,
            'min_ms1_intensity': min_ms1_intensity,
            'min_rt': min_rt,
            'max_rt': max_rt
        }
        return params

    def simulate(self, chems, scan_duration, params):
        run_TopN(chems, scan_duration, params, self.out_dir)

    def evaluate(self, params):
        xml_file = self.config_parser.get('evaluation', 'mzmine_xml_file')
        mzmine_command = self.config_parser.get('evaluation', 'mzmine_command')
        boxes = extract_boxes(self.seed_file, self.out_dir, mzmine_command,
                              xml_file)
        evaluate_boxes_as_dict(boxes, self.out_dir)


class SmartROISimulator(InSilicoSimulator):
    def get_params(self):
        # get experiment parameters
        ionisation_mode = self.config_parser.get('experiment',
                                                 'ionisation_mode')
        isolation_width = self.config_parser.getfloat('experiment',
                                                      'isolation_width')
        min_rt = self.config_parser.getfloat('experiment', 'min_rt')
        max_rt = self.config_parser.getint('experiment', 'max_rt')

        # get simulation parameters
        N = self.config_parser.getint('simulation', 'N')
        mz_tol = self.config_parser.getint('simulation', 'mz_tol')
        rt_tol = self.config_parser.getint('simulation', 'rt_tol')
        min_ms1_intensity = self.config_parser.getint('simulation',
                                                      'min_ms1_intensity')

        # get additional SmartROI parameters
        iif_values = self.config_parser.get('simulation', 'iif_values')
        dp_values = self.config_parser.get('simulation', 'dp_values')
        iif_values = string_to_list(iif_values, convert=float)
        dp_values = string_to_list(dp_values, convert=float)

        min_roi_intensity = self.config_parser.getfloat('simulation',
                                                        'min_roi_intensity')
        min_roi_length = self.config_parser.getint('simulation',
                                                   'min_roi_length')
        min_frag = self.config_parser.getint(
            'simulation', 'min_roi_length_for_fragmentation')

        params = {
            'controller_name': self.controller_name,
            'ionisation_mode': ionisation_mode,
            'sample_name': self.sample_name,
            'isolation_width': isolation_width,
            'N': N,
            'mz_tol': mz_tol,
            'rt_tol': rt_tol,
            'min_ms1_intensity': min_ms1_intensity,
            'min_rt': min_rt,
            'max_rt': max_rt,
            'iif_values': iif_values,
            'dp_values': dp_values,
            'min_roi_intensity': min_roi_intensity,
            'min_roi_length': min_roi_length,
            'min_roi_length_for_fragmentation': min_frag
        }
        return params

    def simulate(self, chems, scan_duration, params):
        run_SmartROI(chems, scan_duration, params, self.out_dir)

    def evaluate(self, params):
        # extract peak boxes
        xml_file = self.config_parser.get('evaluation', 'mzmine_xml_file')
        mzmine_command = self.config_parser.get('evaluation', 'mzmine_command')
        boxes = extract_boxes(self.seed_file, self.out_dir, mzmine_command,
                              xml_file)

        # extract counts
        pattern = 'SMART_{}_{}_{}.mzml'
        yticks = params['iif_values']
        xticks = params['dp_values']
        counts = evaluate_boxes_as_array(boxes, self.out_dir, yticks, xticks,
                                         pattern, params)
        save_counts(counts, self.out_dir, params['controller_name'],
                    params['sample_name'])

        # plot counts
        xlabel = r'$\beta$'
        ylabel = r'$\alpha$'
        title = 'SmartROI simulations (%s)' % self.sample_name
        out_file = os.path.join(self.out_dir, '%s_%s.png' % (
            self.controller_name, self.sample_name))
        plot_counts(counts, out_file, title, xlabel, xticks, ylabel, yticks)


class WeightedDEWSimulator(InSilicoSimulator):
    def get_params(self):
        # get experiment parameters
        ionisation_mode = self.config_parser.get('experiment',
                                                 'ionisation_mode')
        isolation_width = self.config_parser.getfloat('experiment',
                                                      'isolation_width')
        min_rt = self.config_parser.getfloat('experiment', 'min_rt')
        max_rt = self.config_parser.getint('experiment', 'max_rt')

        # get simulation parameters
        N = self.config_parser.getint('simulation', 'N')
        mz_tol = self.config_parser.getint('simulation', 'mz_tol')
        min_ms1_intensity = self.config_parser.getint('simulation',
                                                      'min_ms1_intensity')

        # get additional SmartROI parameters
        t0_values = self.config_parser.get('simulation', 't0_values')
        rt_tol_values = self.config_parser.get('simulation', 'rt_tol_values')
        t0_values = string_to_list(t0_values, convert=float)
        rt_tol_values = string_to_list(rt_tol_values, convert=float)

        params = {
            'controller_name': self.controller_name,
            'ionisation_mode': ionisation_mode,
            'sample_name': self.sample_name,
            'isolation_width': isolation_width,
            'N': N,
            'mz_tol': mz_tol,
            'min_ms1_intensity': min_ms1_intensity,
            'min_rt': min_rt,
            'max_rt': max_rt,
            't0_values': t0_values,
            'rt_tol_values': rt_tol_values
        }
        return params

    def simulate(self, chems, time_dict, params):
        run_WeightedDEW(chems, time_dict, params, self.out_dir)

    def evaluate(self, params):
        # extract peak boxes
        xml_file = self.config_parser.get('evaluation', 'mzmine_xml_file')
        mzmine_command = self.config_parser.get('evaluation', 'mzmine_command')
        boxes = extract_boxes(self.seed_file, self.out_dir, mzmine_command,
                              xml_file)

        # extract counts
        pattern = 'WeightedDEW_{}_{}_{}.mzml'
        yticks = params['t0_values']
        xticks = params['rt_tol_values']
        counts = evaluate_boxes_as_array(boxes, self.out_dir, yticks, xticks,
                                         pattern, params)
        save_counts(counts, self.out_dir, params['controller_name'],
                    params['sample_name'])

        xlabel = 't0'
        ylabel = 'rt_tol'
        title = 'WeightedDEW simulations (%s)' % self.sample_name
        out_file = os.path.join(self.out_dir, '%s_%s.png' % (
            self.controller_name, self.sample_name))
        plot_counts(counts, out_file, title, xlabel, xticks, ylabel, yticks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='In-silico Optimisation of Fragmentation '
                    'Strategy using ViMMS')
    parser.add_argument('sample_name', type=str)
    parser.add_argument('seed_file', type=str)
    parser.add_argument('out_dir', type=str)
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()

    # parse config file
    config_parser = configparser.RawConfigParser()
    config_file_path = args.config_file
    config_parser.read(config_file_path)

    # set output log file
    controller_name = config_parser.get('experiment', 'controller_name')
    log_file = '%s_%s.log' % (controller_name, args.sample_name)
    log_path = os.path.join(args.out_dir, log_file)
    add_log_file(log_path, 'INFO')

    # run simulation here
    sample_name = args.sample_name
    seed_file = args.seed_file
    out_dir = args.out_dir
    choices = {
        IN_SILICO_OPTIMISE_TOPN: TopNSimulator(sample_name, seed_file, out_dir,
                                               controller_name, config_parser),
        IN_SILICO_OPTIMISE_SMART_ROI: SmartROISimulator(sample_name, seed_file,
                                                        out_dir,
                                                        controller_name,
                                                        config_parser),
        IN_SILICO_OPTIMISE_WEIGHTED_DEW: WeightedDEWSimulator(sample_name,
                                                              seed_file,
                                                              out_dir,
                                                              controller_name,
                                                              config_parser),
    }
    sim = choices[controller_name]
    sim.run()
