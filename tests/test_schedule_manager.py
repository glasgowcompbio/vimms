from pathlib import Path

import pytest

from vimms.SequenceManager import *
from vimms.Chemicals import ChemicalCreator


### define some useful constants ###

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(Path(DIR_PATH, 'fixtures'))
HMDB = load_obj(Path(BASE_DIR, 'hmdb_compounds.p'))
OUT_DIR = Path(DIR_PATH, 'results')

ROI_SOURCES = [str(Path(BASE_DIR, 'beer_t10_simulator_files'))]
MIN_MS1_INTENSITY = 1
RT_RANGE = [(0, 1200)]
CENTRE_RANGE = 600
MIN_RT = RT_RANGE[0][0]
MAX_RT = RT_RANGE[0][1]
MZ_RANGE = [(0, 1050)]
N_CHEMS = 10

BEER_CHEMS = load_obj(Path(BASE_DIR, 'QCB_22May19_1.p'))
BEER_MIN_BOUND = 550
BEER_MAX_BOUND = 650

MZML_FILE = Path(BASE_DIR, 'small_mzml.mzML')


@pytest.fixture(scope="module")
def fragscan_ps():
    return load_obj(Path(BASE_DIR, 'peak_sampler_mz_rt_int_beerqcb_fragmentation.p'))


@pytest.fixture(scope="module")
def fragscan_dataset_spectra(fragscan_ps):
    chems = ChemicalCreator(fragscan_ps, ROI_SOURCES, HMDB)
    return chems.sample(MZ_RANGE, RT_RANGE, MIN_MS1_INTENSITY, N_CHEMS, 2,
                        get_children_method=GET_MS2_BY_SPECTRA)


class TestScheduleManager:
    """
    Tests the Schedule Manager starting from both a dataset and an mzml file
    """
    def test_schedulemanager_dataset(self):
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
                            'scan_duration_dict': DEFAULT_SCAN_TIME_DICT}

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

        MZML2CHEMS_DICT = {'min_ms1_intensity': 1.75E5,
                           'mz_tol': 5,
                           'mz_units': 'ppm',
                           'min_length': 1,
                           'min_intensity': 0,
                           'start_rt': 0,
                           'stop_rt': 1560,
                           'n_peaks': 1}

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
                            'scan_duration_dict': DEFAULT_SCAN_TIME_DICT}

        d2 = {
            'Sample ID': ['blank1', 'sample1', 'blank2', 'sample2'],
            'Controller Method': [None, 'TopNController', None, 'TopNController'],
            'Controller Params': [None, controller_params, None, controller_params],
            'MassSpec Params': [None, mass_spec_params, None, mass_spec_params],
            'Dataset': [None, None, None, None]
        }
        controller_schedule2 = pd.DataFrame(data=d2)

        mzml_file_list = [None, MZML_FILE, None, MZML_FILE]

        vsm = VimmsSequenceManager(controller_schedule2, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=True, mzmine_command=None)
        experiment = BasicExperiment(vsm, parallel=False, mzml_file_list=mzml_file_list,
                                     MZML2CHEMS_DICT=MZML2CHEMS_DICT, ps=fragscan_ps)


class TestGridSearch:
    """
    Tests the Grid Search starting from both a dataset and an mzml file
    """
    def test_gridsearch_dataset(self):
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
                            'scan_duration_dict': DEFAULT_SCAN_TIME_DICT}

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
                            'scan_duration_dict': DEFAULT_SCAN_TIME_DICT}

        MZML2CHEMS_DICT = {'min_ms1_intensity': 1.75E5,
                           'mz_tol': 5,
                           'mz_units': 'ppm',
                           'min_length': 1,
                           'min_intensity': 0,
                           'start_rt': 0,
                           'stop_rt': 1560,
                           'n_peaks': 1}

        vsm = VimmsSequenceManager(None, evaluation_methods, OUT_DIR, ms1_picked_peaks_file=None,
                                   progress_bar=False, mzmine_command=None)
        gs = GridSearchExperiment(vsm, 'TopNController', mass_spec_params, None, topn_variable_params_dict,
                                  controller_params, MZML_FILE, MZML2CHEMS_DICT=MZML2CHEMS_DICT, ps=fragscan_ps,
                                  parallel=False)

