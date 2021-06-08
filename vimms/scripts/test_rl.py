import sys

sys.path.append('..')
sys.path.append('../..')  # if running in this folder

import torch
import random as rand
from gym import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from vimms.Common import *
from vimms.Gym import VimmsGymEnv
from vimms.ChemicalSamplers import MZMLFormulaSampler, MZMLRTandIntensitySampler
from vimms.Controller import SimpleTargetController

np.random.seed(0)
rand.seed(0)
torch.manual_seed(0)

set_log_level_warning()

n_chemicals = (200, 500)
mz_range = (100, 600)
rt_range = (200, 1000)
intensity_range = (1E5, 1E10)

min_mz = mz_range[0]
max_mz = mz_range[1]
min_rt = rt_range[0]
max_rt = rt_range[1]
min_log_intensity = np.log(intensity_range[0])
max_log_intensity = np.log(intensity_range[1])

isolation_window = 0.7
N = 10
rt_tol = 15
mz_tol = 10
min_ms1_intensity = 5000
ionisation_mode = POSITIVE
noise_density = 0.3
noise_max_val = 1e4

in_dim = 50
out_dim = 50

mzml_filename = os.path.abspath(os.path.join('..', '..', 'experimental', 'Beer_multibeers_1_fullscan1.mzML'))
mz_sampler = MZMLFormulaSampler(mzml_filename, min_mz=min_mz, max_mz=max_mz)
ri_sampler = MZMLRTandIntensitySampler(mzml_filename, min_rt=min_rt, max_rt=max_rt,
                                       min_log_intensity=min_log_intensity,
                                       max_log_intensity=max_log_intensity)

params = {
    'chemical_creator': {
        'mz_range': mz_range,
        'rt_range': rt_range,
        'intensity_range': intensity_range,
        'n_chemicals': n_chemicals,
        'mz_sampler': mz_sampler,
        'ri_sampler': ri_sampler,
    },
    'noise': {
        'noise_density': noise_density,
        'noise_max_val': noise_max_val,
        'mz_range': mz_range
    },
    'env': {
        'ionisation_mode': ionisation_mode,
        'rt_range': rt_range,
        'N': N,
        'isolation_window': isolation_window,
        'mz_tol': mz_tol,
        'rt_tol': rt_tol,
        'min_ms1_intensity': min_ms1_intensity
    }
}


class MaxIntensityEnv(VimmsGymEnv):
    def __init__(self, in_dim, out_dim, params):
        super().__init__(in_dim, out_dim, params)
        self.last_excluded = []
        self.selected_precursors = []

    def _get_action_space(self):
        """
        Defines action space
        """
        return spaces.MultiBinary(self.out_dim)

    def _get_observation_space(self):
        """
        Defines observation space
        """
        features = {
            'intensity': spaces.Box(low=0.0, high=np.inf, shape=(self.in_dim,)),
            'exclusion': spaces.Box(low=0.0, high=1.0, shape=(self.in_dim,))
        }
        return spaces.Dict(features)

    def _get_state(self, scan_to_process):
        """
        Converts a scan to a state
        """
        self.last_scan_to_process = scan_to_process
        mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)

        precursors = []
        self.last_excluded = []
        for mz, intensity in zip(mzs, intensities):
            if self.controller.exclusion.is_excluded(mz, rt):
                excluded = 1
                self.last_excluded.append((mz, rt, intensity,))
            else:
                excluded = 0
            precursor = (mz, intensity, excluded,)
            precursors.append(precursor)

        sorted_precursors = sorted(precursors, key=lambda item: item[1], reverse=True)  # sort by intensity descending
        self.selected_precursors = []
        feature_intensity = []
        feature_exclusion = []
        for i in range(self.in_dim):  # get the first in_dim items
            precursor = sorted_precursors[i]
            mz, intensity, excluded = precursor
            self.selected_precursors.append(precursor)

            intensity = np.log(intensity)
            if np.isnan(intensity):
                intensity = 0
            feature_intensity.append(intensity)
            feature_exclusion.append(excluded)

        feature_intensity = np.array(feature_intensity)
        feature_exclusion = np.array(feature_exclusion)
        features = {
            'intensity': feature_intensity,
            'exclusion': feature_exclusion
        }
        return features

    def _compute_reward(self, scan_to_process, results):
        """
        Computes fragmentation reward
        """
        parent_scan_id = self.controller.last_ms1_scan.scan_id
        assert scan_to_process.scan_id == parent_scan_id

        total_reward = 0.0
        for last_scan in results:
            if last_scan.ms_level >= 2:
                precursor = last_scan.scan_params.get(ScanParameters.PRECURSOR_MZ)[0]
                mz = precursor.precursor_mz
                frag_rt = last_scan.scan_params.get(ScanParameters.METADATA)['frag_at']
                intensity = precursor.precursor_intensity
                time_filter = -1 if (mz, frag_rt, intensity, ) in self.last_excluded else 1
                reward = np.log(intensity) * time_filter
                total_reward += reward
        return total_reward

    def _take_action(self, action):
        """
        Modify controller variables based on the selected action
        """
        target_flag = action
        assert len(target_flag) == len(self.selected_precursors)

        targets = []
        for i in range(len(target_flag)):
            t = target_flag[i]
            if t == 1:
                mz, intensity, excluded = self.selected_precursors[i]
                targets.append((mz, intensity))

        # self.seen_actions.update(['targets=%s' % targets])
        self.controller.targets = targets

    def _reset_controller(self, env_params):
        """
        Generates new controller
        """
        ionisation_mode = env_params['ionisation_mode']
        N = env_params['N']
        isolation_window = env_params['isolation_window']
        mz_tol = env_params['mz_tol']
        rt_tol = env_params['rt_tol']
        min_ms1_intensity = env_params['min_ms1_intensity']
        controller = SimpleTargetController(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)
        return controller


set_log_level_info()

env = MaxIntensityEnv(in_dim, out_dim, params)
check_env(env)

model_name = 'PPO'
model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=0.01,
            tensorboard_log='./results/%s_MaxIntensityEnv_tensorboard' % model_name)
model.learn(total_timesteps=10000)

fname = 'results/%s_maxintensity_smallchems' % model_name
model.save(fname)
