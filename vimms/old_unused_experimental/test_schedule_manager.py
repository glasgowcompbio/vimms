from pathlib import Path

import pandas as pd

from tests.conftest import BASE_DIR, OUT_DIR, MZML_FILE
from vimms.Common import POSITIVE, DEFAULT_SCAN_TIME_DICT
from vimms.old_unused_experimental.SequenceManager import VimmsSequenceManager, BasicExperiment, GridSearchExperiment


class TestScheduleManager:
    """
    Tests the Schedule Manager starting from both a dataset and an mzml file
    """

    def test_schedulemanager_dataset(self, fragscan_ps):
        evaluation_methods = []

        dataset_file = Path(BASE_DIR, 'QCB_22May19_1.p')

        controller_params = {"ionisation_mode": POSITIVE,
                             "N": 10,
                             "mz_tol": 10,
                             "rt_tol": 30,
                             "min_ms1_intensity": 1.75E5,
                             "rt_range": [(200, 400)],
                             "isolation_width": 1}

        mass_spec_params = {'ionisation_mode': POSITIVE,
                            'peak_sampler': fragscan_ps,
                            'mz_noise': None,
                            'intensity_noise': None,
                            'isolation_transition_window': 'rectangular',
                            'isolation_transition_window_params': None,
                            'scan_duration': DEFAULT_SCAN_TIME_DICT}

        d = {
            'Sample ID': ['blank1', 'sample1', 'blank2', 'sample2'],
            'Controller Method': [None, 'TopNController', None, 'TopNController'],
            'Controller Params': [None, controller_params, None, controller_params],
            'MassSpec Params': [None, mass_spec_params, None, mass_spec_params],
            'Dataset': [None, dataset_file, None, dataset_file]
        }
        controller_schedule = pd.DataFrame(data=d)

        vsm = VimmsSequenceManager(controller_schedule, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=False, mzmine_command=None)
        experiment = BasicExperiment(vsm, parallel=False)

    def test_schedulemanager_mzml(self, fragscan_ps):
        evaluation_methods = []

        controller_params = {"ionisation_mode": POSITIVE,
                             "N": 10,
                             "mz_tol": 10,
                             "rt_tol": 30,
                             "min_ms1_intensity": 1.75E5,
                             "rt_range": [(200, 400)],
                             "isolation_width": 1}

        mass_spec_params = {'ionisation_mode': POSITIVE,
                            'peak_sampler': fragscan_ps,
                            'mz_noise': None,
                            'intensity_noise': None,
                            'isolation_transition_window': 'rectangular',
                            'isolation_transition_window_params': None,
                            'scan_duration': DEFAULT_SCAN_TIME_DICT}

        d2 = {
            'Sample ID': ['blank1', 'sample1', 'blank2', 'sample2'],
            'Controller Method': [None, 'TopNController', None, 'TopNController'],
            'Controller Params': [None, controller_params, None, controller_params],
            'MassSpec Params': [None, mass_spec_params, None, mass_spec_params],
            'Dataset': [None, None, None, None]
        }
        controller_schedule2 = pd.DataFrame(data=d2)

        mzml_file_list = [None, str(MZML_FILE), None, str(MZML_FILE)]

        vsm = VimmsSequenceManager(controller_schedule2, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=True, mzmine_command=None)
        experiment = BasicExperiment(vsm, parallel=False, mzml_file_list=mzml_file_list, ps=fragscan_ps)


class TestGridSearch:
    """
    Tests the Grid Search starting from both a dataset and an mzml file
    """

    def test_gridsearch_dataset(self, fragscan_ps):
        evaluation_methods = []
        topn_variable_params_dict = {'N': [10], 'rt_tol': [15, 30]}

        dataset_file = Path(BASE_DIR, 'QCB_22May19_1.p')

        controller_params = {"ionisation_mode": POSITIVE,
                             "N": 10,
                             "mz_tol": 10,
                             "rt_tol": 30,
                             "min_ms1_intensity": 1.75E5,
                             "rt_range": [(200, 400)],
                             "isolation_width": 1}

        mass_spec_params = {'ionisation_mode': POSITIVE,
                            'peak_sampler': fragscan_ps,
                            'mz_noise': None,
                            'intensity_noise': None,
                            'isolation_transition_window': 'rectangular',
                            'isolation_transition_window_params': None,
                            'scan_duration': DEFAULT_SCAN_TIME_DICT}

        vsm = VimmsSequenceManager(None, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=False, mzmine_command=None)
        gs = GridSearchExperiment(vsm, 'TopNController', mass_spec_params, dataset_file, topn_variable_params_dict,
                                  controller_params, parallel=False)

    def test_gridsearch_mzml(self, fragscan_ps):
        evaluation_methods = []
        topn_variable_params_dict = {'N': [10], 'rt_tol': [15, 30]}

        controller_params = {"ionisation_mode": POSITIVE,
                             "N": 10,
                             "mz_tol": 10,
                             "rt_tol": 30,
                             "min_ms1_intensity": 1.75E5,
                             "rt_range": [(200, 400)],
                             "isolation_width": 1}

        mass_spec_params = {'ionisation_mode': POSITIVE,
                            'peak_sampler': fragscan_ps,
                            'mz_noise': None,
                            'intensity_noise': None,
                            'isolation_transition_window': 'rectangular',
                            'isolation_transition_window_params': None,
                            'scan_duration': DEFAULT_SCAN_TIME_DICT}

        vsm = VimmsSequenceManager(None, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=False, mzmine_command=None)
        gs = GridSearchExperiment(vsm, 'TopNController', mass_spec_params, None, topn_variable_params_dict,
                                  controller_params, str(MZML_FILE), ps=fragscan_ps,
                                  parallel=False)
