import glob
import inspect
import itertools
import time
from os.path import dirname, abspath
from pathlib import Path

import seaborn as sns
from mass_spec_utils.data_import.mzmine import map_boxes_to_scans
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.data_processing.alignment import BoxJoinAligner
from sklearn.linear_model import LogisticRegression

from vimms.Controller import *
from vimms.Environment import Environment
from vimms.FeatureExtraction import extract_roi
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.PythonMzmine import pick_peaks
from vimms.Scoring import picked_peaks_evaluation, roi_scoring

parent_dir = dirname(dirname(abspath(__file__)))
batch_file_dir = os.path.join(parent_dir, 'batch_files')
QCB_XML_TEMPLATE_MS1 = os.path.join(batch_file_dir, 'QCB_mzmine_batch_ms1.xml')
QCB_XML_TEMPLATE_MS2 = os.path.join(batch_file_dir, 'QCB_mzmine_batch_ms2.xml')

MZMINE_COMMAND = 'C:\\Users\\joewa\\Work\\git\\MZmine-2.40.1\\startMZmine_Windows.bat'


class BaseSequenceManager(object):
    def __init__(self, controller_schedule, evaluation_methods, base_dir,
                 evaluaton_min_ms1_intensity=1.75E5,
                 evaluation_params=None,
                 ms1_picked_peaks_file=None,
                 align_peaks=False,
                 xml_template_ms1=QCB_XML_TEMPLATE_MS1,
                 xml_template_ms2=QCB_XML_TEMPLATE_MS2,
                 mzmine_command=MZMINE_COMMAND,
                 progress_bar=False,
                 write_env=False,
                 rt_range=[(0, 1440)]):

        logger.debug('evaluation_methods: %s' % evaluation_methods)
        logger.debug('base_dir: %s' % base_dir)
        create_if_not_exist(base_dir)

        logger.debug('xml_template_ms1: %s' % xml_template_ms1)
        logger.debug('xml_template_ms2: %s' % xml_template_ms2)
        logger.debug('mzmine_command: %s' % mzmine_command)

        self.controller_schedule = controller_schedule

        # above needs to include controller type, controller params, MS stuff, Case-Control status
        self.evaluation_methods = evaluation_methods
        self.base_dir = base_dir
        self.evaluation_min_ms1_intensity = evaluaton_min_ms1_intensity
        self.evaluation_params = evaluation_params
        self.ms1_picked_peaks_file = ms1_picked_peaks_file
        self.align_peaks = align_peaks
        self.xml_template_ms1 = xml_template_ms1
        self.xml_template_ms2 = xml_template_ms2
        self.mzmine_command = mzmine_command
        self.progress_bar = progress_bar
        self.write_env = write_env
        self.rt_range = rt_range

    def run(self):
        NotImplementedError()

    def run_experiment(self, idx):
        # runs the INDIVIDUAL experiments as determined by extended classes
        ms_params = self.controller_schedule['MassSpec Params'][idx]
        controller = create_controller(self.controller_schedule['Controller Method'][idx],
                                       self.controller_schedule['Controller Params'][idx])
        return controller, ms_params

    def run_parallel(self):
        NotImplementedError()

    def add_to_evaluation_queue(self):
        NotImplementedError()

    def evaluate_controller(self, ms2_mzml_file):
        if 'mzmine_peak' in self.evaluation_methods:
            mzmine_peaks_score = picked_peaks_evaluation(ms2_mzml_file, self.ms1_picked_peaks_file)
            logger.debug(mzmine_peaks_score)
            mzmine_peaks_dict = {'mzmine_peak': mzmine_peaks_score}
        else:
            mzmine_peaks_dict = {'mzmine_peak': 0}
        if 'roi_coverage' in self.evaluation_methods:
            roi_scores = roi_scoring(ms2_mzml_file)
            roi_score_dict = {'roi_coverage': roi_scores['with_scan']}
        else:
            roi_score_dict = {'roi_coverage': 0}
        scores_dict = dict(mzmine_peaks_dict.items() | roi_score_dict.items())
        return scores_dict

    def pick_peaks(self, mzml_file, controller_name, ms_level):
        if ms_level == 1:
            picked_peaks_file = os.path.join(self.base_dir, Path(mzml_file).stem + '_pp.csv')
            pick_peaks([mzml_file], xml_template=self.xml_template_ms1, output_dir=self.base_dir,
                       mzmine_command=self.mzmine_command)
        else:
            picked_peaks_file = self.base_dir + '\\' + controller_name + '_pp.csv'
            pickle_pp = glob.glob(os.path.join(self.base_dir, '*.csv'))
            if os.path.basename(controller_name) + '_pp.csv' not in [os.path.basename(file) for file in pickle_pp]:
                pick_peaks([mzml_file], xml_template=self.xml_template_ms2, output_dir=self.base_dir,
                           mzmine_command=self.mzmine_command)
            else:
                logger.info('Found picked peaks file. Skipping...')
        return picked_peaks_file

    def update_aligned_scores(self):
        NotImplementedError()

    def create_results_df(self):
        # TODO: needs implementing, but not until evaluation is run on a separate thread
        # TODO: evaluation_methods also needs to be set to None until multi thread
        results_df = pd.DataFrame(dict(list(dict(self.controller_schedule.iloc[0, 0:2]).items()) +
                                       list(self.controller_schedule['Controller Params'][0].items())), index=[0])
        for i in range(1, len(self.controller_schedule.index)):
            results_df = results_df.append(dict(list(dict(self.controller_schedule.iloc[i, 0:2]).items()) +
                                                list(self.controller_schedule['Controller Params'][i].items())),
                                           ignore_index=True)
        remove_elements = list(set(results_df.columns) & set(['rt_range', 'ionisation_mode', 'isolation_width']))
        results_df = results_df.drop(remove_elements, axis=1)
        if self.evaluation_methods is not None:
            for score_name in self.evaluation_methods:
                results_df[score_name] = [0.0 for i in range(len(self.controller_schedule.index))]
        return results_df


class VimmsSequenceManager(BaseSequenceManager):
    def __init__(self, controller_schedule, evaluation_methods, base_dir,
                 evaluaton_min_ms1_intensity=1.75E5,
                 evaluation_params=None,
                 ms1_picked_peaks_file=None,
                 align_peaks=False,
                 xml_template_ms1=QCB_XML_TEMPLATE_MS1,
                 xml_template_ms2=QCB_XML_TEMPLATE_MS2,
                 mzmine_command=MZMINE_COMMAND,
                 progress_bar=False,
                 write_env=False,
                 rt_range=[(0, 1440)]):
        super().__init__(controller_schedule, evaluation_methods, base_dir, evaluaton_min_ms1_intensity,
                         evaluation_params, ms1_picked_peaks_file, align_peaks, xml_template_ms1,
                         xml_template_ms2, mzmine_command, progress_bar, write_env, rt_range)

        if self.controller_schedule is not None:
            # filter controller_schedule to remove blank and calibration samples
            # in this case, we set the Controller Method to None in the schedule file
            self.schedule_idx = np.where(np.array(controller_schedule['Controller Method']) != None)[0]
            self.controller_schedule = controller_schedule[
                controller_schedule['Controller Method'].values != None].reset_index(drop=True)

    def run_experiment(self, idx):
        controller_name = self.controller_schedule['Sample ID'][idx]
        mzml_files = glob.glob(os.path.join(self.base_dir, '*.mzML'))
        if controller_name + '.mzML' not in [os.path.basename(file) for file in mzml_files]:
            controller, ms_params = super().run_experiment(idx)
            # load data and set up MS
            logger.info(self.controller_schedule.iloc[[idx]].to_dict())
            method = self.controller_schedule['Controller Method'][idx]
            dataset = self.controller_schedule['Dataset'][idx]
            if method is not None and dataset is not None:
                dataset = load_obj(self.controller_schedule['Dataset'][idx])
                mass_spec = IndependentMassSpectrometer(ms_params['ionisation_mode'], dataset, ms_params['peak_sampler'],
                                                        ms_params['add_noise'], ms_params['isolation_transition_window'],
                                                        ms_params['isolation_transition_window_params'])
                # Run sample
                env = Environment(mass_spec, controller, self.rt_range[0][0],
                                  self.rt_range[0][1], progress_bar=self.progress_bar)
                env.run()
                env.write_mzML(self.base_dir, controller_name + '.mzML')
                if self.write_env:
                    save_obj(controller, os.path.join(self.base_dir, controller_name + '.p'))
        else:
            logger.info('Experiment already completed. Skipping...')
        mzml_file = os.path.join(self.base_dir, controller_name + '.mzML')
        return mzml_file, controller_name


########################################################################################################################
# Experiments
########################################################################################################################

class Experiment(object):
    def __init__(self, sequence_manager):
        self.sequence_manager = sequence_manager
        self.results = self.sequence_manager.create_results_df()
        self.run()

    def run(self):
        NotImplementedError()

    def run_controller(self, idx):
        logger.info('Begun experiment: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        if self.sequence_manager.progress_bar:
            time.sleep(0.2)  # visual effect in case progress_bar = True
        current_mzml_file, controller_name = self.sequence_manager.run_experiment(idx)  # runs the idx th experiment
        logger.info('Completed experiment: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        return current_mzml_file, controller_name

    def run_peak_picking(self, idx, current_mzml_file, controller_name):
        logger.info('Started Peak Picking: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        ms2_picked_peaks_file = self.sequence_manager.pick_peaks(current_mzml_file, controller_name, 2)
        logger.info('Completed Peak Picking: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        return ms2_picked_peaks_file

    def run_evaluation(self, idx, current_mzml_file, ms2_picked_peaks_file):
        logger.info('Started Evaluation: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        results_df = self.sequence_manager.evaluate_controller(current_mzml_file)
        logger.info('Completed Evaluation: ' + self.sequence_manager.controller_schedule['Sample ID'][idx])
        return results_df


class BasicExperiment(Experiment):
    def __init__(self, sequence_manager, parallel=True, mzml_file_list=None, MZML2CHEMS_DICT=None, ps=None):
        self.parallel = parallel
        self.mzml2chems_dict = MZML2CHEMS_DICT
        self.ps = ps
        sequence_manager = self.add_defaults_controller_params(sequence_manager)
        if mzml_file_list is not None and all(np.array(sequence_manager.controller_schedule['Dataset']) == None):
            sequence_manager = self.add_dataset_files(sequence_manager, mzml_file_list)
        super().__init__(sequence_manager)

    def add_dataset_files(self, sequence_manager, mzml_file_list):
        for i in range(len(sequence_manager.controller_schedule['Dataset'])):
            if mzml_file_list[sequence_manager.schedule_idx[i]] is not None:
                mzml_file = mzml_file_list[sequence_manager.schedule_idx[i]]
                datasets = extract_roi([mzml_file], None, None, None, self.ps, param_dict=self.mzml2chems_dict)
                dataset = datasets[0]
                dataset_name = os.path.join(sequence_manager.base_dir, Path(mzml_file_list[sequence_manager.schedule_idx[i]]).stem + '.p')
                save_obj(dataset, dataset_name)
                sequence_manager.controller_schedule['Dataset'][i] = dataset_name
        return sequence_manager

    def run(self):
        if self.parallel:
            logger.info('Running in parallel mode')
            self.run_parallel()
        else:
            logger.info('Running in serial mode')
            self.run_serial()

    def run_serial(self):
        for idx in range(self.sequence_manager.controller_schedule.shape[0]):
            params = {
                'self': self,
                'idx': idx
            }
            results_df = run_single(params)
            for score_name in self.sequence_manager.evaluation_methods:
                self.results.loc[idx, score_name] = results_df[score_name]
            logger.info('Finished %d' % idx)

    def run_parallel(self):
        import ipyparallel as ipp
        rc = ipp.Client()
        dview = rc[:]  # use all engines​
        with dview.sync_imports():
            pass

        params_list = []
        for idx in range(self.sequence_manager.controller_schedule.shape[0]):
            params = {
                'self': self,
                'idx': idx
            }
            params_list.append(params)

        logger.info(params_list)
        results = dview.map_sync(run_single, params_list)
        for idx in range(len(results)):
            results_df = results[idx]
            for score_name in self.sequence_manager.evaluation_methods:
                self.results.loc[idx, score_name] = results_df[score_name]
            logger.info('Finished %d' % idx)

    def add_defaults_controller_params(self, sequence_manager):
        for i in range(len(sequence_manager.controller_schedule['Controller Params'])):
            if sequence_manager.controller_schedule['Controller Params'][i] is not None:
                new_dict = merge_controller_param_dict(sequence_manager.controller_schedule['Controller Params'][i],
                                                       dict(),
                                                       sequence_manager.controller_schedule['Controller Method'][i])
                sequence_manager.controller_schedule['Controller Params'][i] = new_dict
        return sequence_manager


def run_single(params):
    idx = params['idx']
    self = params['self']

    current_mzml_file, controller_name = self.run_controller(idx)
    # TODO: add skip if calibration run
    if self.sequence_manager.align_peaks:
        ms2_picked_peaks_file = self.run_peak_picking(idx, current_mzml_file, controller_name)
    else:
        ms2_picked_peaks_file = None

    # TODO: do evaluation on a separate thread, so we don't block the run
    if self.sequence_manager.evaluation_methods is not None:
        results_df = self.run_evaluation(idx, current_mzml_file, ms2_picked_peaks_file)
    return results_df


class GridSearchExperiment(BasicExperiment):
    def __init__(self, sequence_manager, controller_method, mass_spec_param_dict, dataset_file, variable_params_dict,
                 base_params_dict, mzml_file=None, MZML2CHEMS_DICT=None, ps=None, parallel=True):

        self.sequence_manager = sequence_manager
        self.parallel = parallel
        self.controller_method = controller_method
        self.mass_spec_param_dict = mass_spec_param_dict
        self.dataset_file = dataset_file
        self.mzml_file = mzml_file
        if self.dataset_file is None:
            dataset = extract_roi([self.mzml_file], None, None, None, ps, param_dict=MZML2CHEMS_DICT)
            dataset_name = os.path.join(sequence_manager.base_dir, Path(mzml_file).stem + '.p')
            save_obj(dataset, dataset_name)
            self.dataset_file = dataset_name
            if self.sequence_manager.ms1_picked_peaks_file is None:
                self.sequence_manager.ms1_picked_peaks_file = self.sequence_manager.pick_peaks(self.mzml_file, None, 1)
        self.variable_params_dict = variable_params_dict
        self.base_params_dict = base_params_dict
        sequence_manager.controller_schedule = self._generate_controller_schedule()
        super().__init__(sequence_manager, self.parallel)

    def run(self):
        super().run()

    def _generate_controller_schedule(self):
        # get controller_dictionaries
        keys, values = zip(*self.variable_params_dict.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        dicts = []
        for exp in experiments:
            dicts.append(merge_controller_param_dict(exp, self.base_params_dict, self.controller_method))
        # create new schedule
        schedule_data = {'Sample ID': ['sample' + str(i) for i in range(len(dicts))],
                         'Controller Method': [self.controller_method for i in range(len(dicts))],
                         'Controller Params': [dicts[i] for i in range(len(dicts))],
                         'MassSpec Params': [self.mass_spec_param_dict for i in range(len(dicts))],
                         'Dataset': [self.dataset_file for i in range(len(dicts))]}
        controller_schedule = pd.DataFrame(data=schedule_data)
        # return schedule
        return controller_schedule


class BayesianOptimisationExperiment(GridSearchExperiment):
    NotImplementedError()
    # start with gridsearch experiment and add the extra stuff
    # or potentially dont and just use an already implemented BO approach - as doesnt need to work on MS


class DsdaExperiment(Experiment):
    def __init__(self, sequence_manager):
        self.aligned_peaks_df = []
        self.scores = []
        # TODO: add check that was have all DsDA controllers or blanks
        # TODO: update DsDA controller to take scores and aligned peaks
        super().__init__(sequence_manager)

    def run(self):
        for idx in range(self.sequence_manager.controller_schedule.shape[0]):
            current_mzml_file, controller_name = self.run_controller(idx)
            ms2_picked_peaks_file = self.run_peak_picking(idx, current_mzml_file, controller_name)
            if self.sequence_manager.evaluation_methods is not None:
                self.run_evaluation(idx, current_mzml_file, ms2_picked_peaks_file)
            self.aligned_peaks_df = self.align_peaks(self.aligned_peaks_df, ms2_picked_peaks_file)
            self.scores = self.get_scores()

    def get_scores(self):
        scores = [1 for i in range(len(self.aligned_peaks_df.index()))]
        return scores


class RepeatedExperiment(Experiment):
    def __init__(self, sequence_manager):
        self.aligned_peaks_df = []
        self.scores = []
        # TODO: add check that was have all DsDA controllers or blanks
        # TODO: update Repeated controller to take scores and aligned peaks
        super().__init__(sequence_manager)

    def run(self):
        for idx in range(self.sequence_manager.controller_schedule.shape[0]):
            current_mzml_file, controller_name = self.run_controller(idx)
            ms2_picked_peaks_file = self.run_peak_picking(idx, current_mzml_file, controller_name)
            if self.sequence_manager.evaluation_methods is not None:
                self.run_evaluation(idx, current_mzml_file, ms2_picked_peaks_file)
            self.aligned_peaks_df = self.align_peaks(self.aligned_peaks_df, ms2_picked_peaks_file)
            self.scores = self.get_scores()

    def get_scores(self):
        scores = [1 for i in range(len(self.aligned_peaks_df.index()))]
        return scores


# TODO: ISSUE - peak picking and alignment is dependent on analysis we run. evaluation may need to run MS1 on all
#  samples, align those boxes, then align them to these - to give true performace


# TODO: maybe change to be able to take a sequence manager with no schedule

class AlignedExperiment(Experiment):
    def __init__(self, sequence_manager, controller_method, mass_spec_param_dict, dataset_files, controller_param_dict,
                 aligner_rt_tolerance=0.5):
        self.sequence_manager = sequence_manager
        self.parallel = False
        self.controller_method = controller_method
        self.mass_spec_param_dict = mass_spec_param_dict
        self.dataset_files = dataset_files
        self.controller_param_dict = controller_param_dict
        self.completed_mzmls = []
        self.aligner = BoxJoinAligner(rt_tolerance=aligner_rt_tolerance)
        self.experiment_scores = []
        self.experiment_ids = []
        self.peak_boxes = []
        self.sequence_manager.controller_schedule = self._generate_controller_schedule()
        self.run()

    def run(self):
        for idx in range(self.sequence_manager.controller_schedule.shape[0]):
            self.run_single(idx)

    def run_single(self, idx):
        current_mzml_file, controller_name = self.run_controller(idx)
        self.completed_mzmls.append(current_mzml_file)
        ms2_picked_peaks_file = self.run_peak_picking(idx, current_mzml_file, controller_name)
        self.aligner = self.run_alignment(self.aligner, ms2_picked_peaks_file)
        self.experiment_scores, self.experiment_ids, self.peak_boxes = self.update_scores(self.aligner,
                                                                         self.experiment_ids,
                                                                         self.experiment_scores,
                                                                         current_mzml_file)

    def _generate_controller_schedule(self):
        sample_ids = ['sample' + str(i) for i in range(len(self.dataset_files))]
        controller_methods = [self.controller_method for i in range(len(self.dataset_files))]
        updated_controller_params = merge_controller_param_dict(self.controller_param_dict, {}, self.controller_method)
        controller_params = [updated_controller_params for i in range(len(self.dataset_files))]
        mass_spec_params = [self.mass_spec_param_dict for i in range(len(self.dataset_files))]
        schedule_data = {'Sample ID': sample_ids,
                         'Controller Method': controller_methods,
                         'Controller Params': controller_params,
                         'MassSpec Params': mass_spec_params,
                         'Dataset': self.dataset_files}
        controller_schedule = pd.DataFrame(data=schedule_data)
        return controller_schedule


    def run_alignment(self, aligner, picked_peak_file):
        aligner.add_file(picked_peak_file)
        return aligner

    def update_scores(self, aligner, experiment_ids, experiment_scores, ms2_mzml):
        # get boxes
        boxes = [aligner.peaksets2boxes[peakset] for peakset in aligner.peaksets]
        # get box ids
        box_ids = [box.peak_id for box in boxes]
        # get box scores
        mz_file = MZMLFile(ms2_mzml)  # TODO: change this to take last of list of mzmls
        scans2boxes, boxes2scans = map_boxes_to_scans(mz_file, boxes, 0.75)  # TODO: add default parameter here
        box_scores = [(box in boxes2scans) * 1 for box in boxes]
        # store results for experiment
        experiment_scores.append(box_scores)
        experiment_ids.append(box_ids)
        if len(experiment_scores) > 1:
            # update old experiment results
            for j in range(len(experiment_scores) - 1):
                updated_experiment_ids = box_ids
                updated_experiment_scores = np.array([0 for id in box_ids])
                for id in updated_experiment_ids:
                    if id in experiment_ids[j]:
                        where1 = np.where(np.array(updated_experiment_ids) == id)[0][0]
                        where2 = np.where(np.array(experiment_ids[j]) == id)[0][0]
                        updated_experiment_scores[where1] = np.array(experiment_scores[j])[where2]
                updated_experiment_scores = list(updated_experiment_scores)
                experiment_scores[j] = updated_experiment_scores
                experiment_ids[j] = updated_experiment_ids
        # get cumulative fragmentation and score
        cumulative_score = []
        for i in range(len(experiment_scores)):
            updated_fragmentation = [max(x) for x in zip(*self.experiment_scores[:i+1])]
            cumulative_score.append(sum(updated_fragmentation))
        self.cumulative_fragmentation = updated_fragmentation
        self.cumulative_score = cumulative_score
        # calculate updated intensities for each box
        self.box_intensities = self.calculate_box_intensities(self.completed_mzmls, boxes)
        return experiment_scores, experiment_ids, boxes

    def calculate_box_intensities(self, mzml_list, boxes):
        intensities = []
        for i in range(len(mzml_list)):
            int, mzs = get_box_intensity(mzml_list[i], boxes)
            intensities.append(int)  # TODO: can be sped up to only update new boxes
        return intensities


class NaiveMultiSampleExperiment(AlignedExperiment):
    def __init__(self, sequence_manager, controller_method, mass_spec_param_dict, dataset_files, controller_param_dict,
                 aligner_rt_tolerance=0.5):
        super().__init__(sequence_manager, controller_method, mass_spec_param_dict, dataset_files,
                         controller_param_dict, aligner_rt_tolerance)


class OptimalMultiSampleExperiment(AlignedExperiment):
    def __init__(self, sequence_manager, controller_method, mass_spec_param_dict, dataset_files, controller_param_dict,
                 aligner_rt_tolerance=0.5):
        super().__init__(sequence_manager, controller_method, mass_spec_param_dict, dataset_files,
                         controller_param_dict, aligner_rt_tolerance)

    def run_single(self, idx):
        if idx > 0:
            self.sequence_manager.controller_schedule['Controller Params'][idx]['peak_boxes'] = self.peak_boxes
            self.sequence_manager.controller_schedule['Controller Params'][idx]['peak_box_scores'] = self.cumulative_fragmentation
            print(self.cumulative_fragmentation)
        super().run_single(idx)


class DifferentialExpressionMultiSampleExperiment(OptimalMultiSampleExperiment):
    def __init__(self, sequence_manager, controller_method, mass_spec_param_dict, dataset_files, controller_param_dict,
                 case_control_status, aligner_rt_tolerance=0.5):
        self.case_control_status = np.array(case_control_status)
        super().__init__(sequence_manager, controller_method, mass_spec_param_dict, dataset_files,
                         controller_param_dict, aligner_rt_tolerance)

    def run_single(self, idx):
        completed_experiments = self.case_control_status[:idx]
        print('completed_experiments', completed_experiments)
        if 'Case' in completed_experiments and 'Control' in completed_experiments:
            model_scores = self.get_model_scores(idx)
            self.sequence_manager.controller_schedule['Controller Params'][idx]['model_scores'] = model_scores
        super().run_single(idx)

    def get_model_scores(self, idx):
        model = LogisticRegression(random_state=0, fit_intercept=True)
        y = np.array([1 for i in range(len(self.case_control_status[:idx + 1]))])
        y[np.where(np.array(self.case_control_status[:idx + 1]) == 'Control')] = 0
        print(y)
        print(len(self.box_intensities))
        model.fit(np.array(np.array(self.box_intensities)), y)
        model_scores = np.abs(model.coef_[0]) + 1
        return model_scores


class CompletedExperiment(BasicExperiment):
    def __init__(self, sequence_manager):
        self.sequence_manager = sequence_manager
        file_names = glob.glob(os.path.join(self.sequence_manager.base_dir, '*.mzML'))
        self.sequence_manager.controller_schedule = self._create_controller_scehdule(file_names)
        self.results = self.create_results_df()
        # create df here
        self.run()

    def run(self):
        super.run()

    def create_controller_schedule(self, file_names):
        sample_names = [Path(file_names[i]).stem for i in range(len(file_names))]
        controller_schedule = pd.DataFrame({'Sample ID': sample_names})
        return controller_schedule

    def create_results_df(self):
        results_df = self.sequence_manager.controller_schedule
        if self.sequence_manager.evaluation_methods is not None:
            for score_name in self.sequence_manager.evaluation_methods:
                results_df[score_name] = [0.0 for i in range(len(self.sequence_manager.controller_schedule.index))]
        return results_df


########################################################################################################################
# Creating Controllers
########################################################################################################################


POSSIBLE_CONTROLLER_DICT = {'TopN_RoiController': TopNController,
                            'DsDA_RoiController': DsDA_RoiController,
                            'TopNController': TopNController,
                            'PurityController': PurityController,
                            'TopN_SmartRoiController': TopN_SmartRoiController,
                            'Repeated_SmartRoiController': Repeated_SmartRoiController,
                            'CaseControl_SmartRoiController': CaseControl_SmartRoiController,
                            'WeightedDewController': ExcludingTopNController}


def merge_controller_param_dict(dict1, dict2, method, possible_controller_dict=POSSIBLE_CONTROLLER_DICT):
    possible_params = inspect.getfullargspec(possible_controller_dict[method]).args[1:]
    defaults = inspect.getfullargspec(possible_controller_dict[method]).defaults
    if defaults is not None:
        all_defaults = np.array([np.nan for i in range(len(possible_params) - len(defaults))] + list(defaults))
    # create blank dictionary
    param_dict = {}
    for param_idx in range(len(possible_params)):
        if possible_params[param_idx] in dict1:
            param_dict[possible_params[param_idx]] = dict1[possible_params[param_idx]]
        elif possible_params[param_idx] in dict2:
            param_dict[possible_params[param_idx]] = dict2[possible_params[param_idx]]
        elif not np.isnan(all_defaults[param_idx]):
            param_dict[possible_params[param_idx]] = all_defaults[param_idx]
        else:
            logger.warning('Not all parameters provided')
    return param_dict


def create_controller(controller_method, param_dict):
    if controller_method == 'TopN_RoiController':
        controller = TopN_RoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                        param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                        param_dict['min_roi_intensity'], param_dict['min_roi_length'], param_dict['N'],
                                        param_dict['rt_tol'], param_dict['min_roi_length_for_fragmentation'],
                                        param_dict['length_units'], param_dict['ms1_shift'],
                                        param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                        param_dict['ms1_collision_energy'], param_dict['ms1_orbitrap_resolution'],
                                        param_dict['ms2_agc_target'], param_dict['ms2_max_it'],
                                        param_dict['ms2_collision_energy'], param_dict['ms2_orbitrap_resolution'])

    if controller_method == 'TopN_SmartRoiController':
        controller = TopN_SmartRoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                             param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                             param_dict['min_roi_intensity'], param_dict['min_roi_length'],
                                             param_dict['N'], param_dict['rt_tol'],
                                             param_dict['min_roi_length_for_fragmentation'],
                                             param_dict['reset_length_seconds'],
                                             param_dict['intensity_increase_factor'], param_dict['length_units'],
                                             param_dict['drop_perc'], param_dict['ms1_shift'],
                                             param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                             param_dict['ms1_collision_energy'], param_dict['ms1_orbitrap_resolution'],
                                             param_dict['ms2_agc_target'], param_dict['ms2_max_it'],
                                             param_dict['ms2_collision_energy'], param_dict['ms2_orbitrap_resolution'])

    elif controller_method == "Repeated_SmartRoiController":
        controller = Repeated_SmartRoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                                 param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                                 param_dict['min_roi_intensity'], param_dict['min_roi_length'],
                                                 param_dict['N'], param_dict['rt_tol'],
                                                 param_dict['min_roi_length_for_fragmentation'],
                                                 param_dict['reset_length_seconds'],
                                                 param_dict['intensity_increase_factor'], param_dict['length_units'],
                                                 param_dict['drop_perc'], param_dict['peak_boxes'],
                                                 param_dict['peak_box_scores'], param_dict['box_increase_factor'],
                                                 param_dict['box_decrease_factor'], param_dict['box_mz_tol'],
                                                 param_dict['ms1_shift'], param_dict['ms1_agc_target'],
                                                 param_dict['ms1_max_it'], param_dict['ms1_collision_energy'],
                                                 param_dict['ms1_orbitrap_resolution'], param_dict['ms2_agc_target'],
                                                 param_dict['ms2_max_it'], param_dict['ms2_collision_energy'],
                                                 param_dict['ms2_orbitrap_resolution'])

    elif controller_method == "CaseControl_SmartRoiController":
        controller = CaseControl_SmartRoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                                    param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                                    param_dict['min_roi_intensity'], param_dict['min_roi_length'],
                                                    param_dict['N'], param_dict['rt_tol'],
                                                    param_dict['min_roi_length_for_fragmentation'],
                                                    param_dict['reset_length_seconds'],
                                                    param_dict['intensity_increase_factor'], param_dict['length_units'],
                                                    param_dict['drop_perc'], param_dict['peak_boxes'],
                                                    param_dict['peak_box_scores'], param_dict['box_increase_factor'],
                                                    param_dict['box_decrease_factor'], param_dict['box_mz_tol'],
                                                    param_dict['coef_scale'], param_dict['model_scores'],
                                                    param_dict['ms1_shift'], param_dict['ms1_agc_target'],
                                                    param_dict['ms1_max_it'], param_dict['ms1_collision_energy'],
                                                    param_dict['ms1_orbitrap_resolution'], param_dict['ms2_agc_target'],
                                                    param_dict['ms2_max_it'], param_dict['ms2_collision_energy'],
                                                    param_dict['ms2_orbitrap_resolution'])

    elif controller_method == 'DsDA_RoiController':
        controller = DsDA_RoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                        param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                        param_dict['min_roi_intensity'], param_dict['dsda_scoring_df'],
                                        param_dict['min_roi_length'], param_dict['N'], param_dict['rt_tol'],
                                        param_dict['min_roi_length_for_fragmentation'], param_dict['length_units'],
                                        param_dict['peak_df'], param_dict['peak_scores'], param_dict['ms1_shift'],
                                        param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                        param_dict['ms1_collision_energy'], param_dict['ms1_orbitrap_resolution'],
                                        param_dict['ms2_agc_target'], param_dict['ms2_max_it'],
                                        param_dict['ms2_collision_energy'], param_dict['ms2_orbitrap_resolution'])

    elif controller_method == 'Probability_RoiController':
        controller = Probability_RoiController(param_dict['ionisation_mode'], param_dict['isolation_width'],
                                               param_dict['mz_tol'], param_dict['min_ms1_intensity'],
                                               param_dict['min_roi_intensity'], param_dict['probability_method'],
                                               param_dict['model_params'], param_dict['min_roi_length'],
                                               param_dict['N'], param_dict['rt_tol'],
                                               param_dict['min_roi_length_for_fragmentation'],
                                               param_dict['length_units'], param_dict['ms1_shift'],
                                               param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                               param_dict['ms1_collision_energy'],
                                               param_dict['ms1_orbitrap_resolution'], param_dict['ms2_agc_target'],
                                               param_dict['ms2_max_it'], param_dict['ms2_collision_energy'],
                                               param_dict['ms2_orbitrap_resolution'])

    elif controller_method == 'TopNController':
        controller = TopNController(param_dict['ionisation_mode'], param_dict['N'], param_dict['isolation_width'],
                                    param_dict['mz_tol'], param_dict['rt_tol'], param_dict['min_ms1_intensity'],
                                    param_dict['ms1_shift'], param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                    param_dict['ms1_collision_energy'], param_dict['ms1_orbitrap_resolution'],
                                    param_dict['ms2_agc_target'], param_dict['ms2_max_it'],
                                    param_dict['ms2_collision_energy'], param_dict['ms2_orbitrap_resolution'])

    elif controller_method == 'PurityController':
        controller = PurityController(param_dict['ionisation_mode'], param_dict['N'],
                                      param_dict['scan_param_changepoints'], param_dict['isolation_width'],
                                      param_dict['mz_tol'], param_dict['rt_tol'], param_dict['min_ms1_intensity'],
                                      param_dict['n_purity_scans'], param_dict['purity_shift'],
                                      param_dict['purity_threshold'], param_dict['purity_randomise'],
                                      param_dict['purity_add_ms1'], param_dict['ms1_agc_target'],
                                      param_dict['ms1_max_it'], param_dict['ms1_collision_energy'],
                                      param_dict['ms1_orbitrap_resolution'], param_dict['ms2_agc_target'],
                                      param_dict['ms2_max_it'], param_dict['ms2_collision_energy'],
                                      param_dict['ms2_orbitrap_resolution'])
    elif controller_method == 'WeightedDewController':
        controller = ExcludingTopNController(param_dict['ionisation_mode'], param_dict['N'],
                                             param_dict['isolation_width'], param_dict['mz_tol'], param_dict['rt_tol'],
                                             param_dict['min_ms1_intensity'], param_dict['ms1_shift'],
                                             param_dict['exclusion_t_0'], param_dict['log_intensity'],
                                             param_dict['ms1_agc_target'], param_dict['ms1_max_it'],
                                             param_dict['ms1_collision_energy'], param_dict['ms1_orbitrap_resolution'],
                                             param_dict['ms2_agc_target'], param_dict['ms2_max_it'],
                                             param_dict['ms2_collision_energy'], param_dict['ms2_orbitrap_resolution'])
    else:
        logger.warning('Invalid controller_method')
    return controller


def Heatmap_GridSearch(GridSearch_object, outcome_name, X_name, Y_name):
    results = GridSearch_object.results.pivot(X_name, Y_name, outcome_name)
    ax = sns.heatmap(results)

