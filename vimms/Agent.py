# Agent.py
from random import randrange

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributions import Categorical

from vimms.Exclusion import TopNExclusion
from vimms.Common import load_obj, set_log_level_warning, set_log_level_debug, GET_MS2_BY_PEAKS, ADDUCT_DICT_POS_MH, \
    GET_MS2_BY_SPECTRA, POSITIVE, ScanParameters



class AbstractAgent(object):
    def __init__(self):
        self.task_list = []

    def next_tasks(self, scan_to_process, controller, current_task_id):
        raise NotImplementedError

    def update(self, last_scan, controller):
        raise NotImplementedError

    def act(self, scan_to_process):
        raise NotImplementedError


class FullScanAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def next_tasks(self, scan_to_process, controller, current_task_id):
        new_tasks = []
        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        current_task_id += 1
        next_processed_scan_id = current_task_id
        self.task_list.append((controller, ms1_scan_params))
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass

    def act(self, scan_to_process):
        pass


class TopNDEWAgent(AbstractAgent):
    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity):
        super().__init__()
        self.ionisation_mode = ionisation_mode
        self.N = N
        self.isolation_width = isolation_width
        self.min_ms1_intensity = min_ms1_intensity
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
        self.exclusion = TopNExclusion()

    def next_tasks(self, scan_to_process, controller, current_task_id):
        self.act(scan_to_process)
        new_tasks, current_task_id, next_processed_scan_id = self._schedule_tasks(controller, current_task_id,
                                                                                  scan_to_process)
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        # update dynamic exclusion list after time has been increased
        self.exclusion.cleanup(last_scan)

    def act(self, scan_to_process):
        pass

    def _schedule_tasks(self, controller, current_task_id, scan_to_process):
        new_tasks = []
        fragmented_count = 0
        mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)

        # loop over points in decreasing intensity
        idx = np.argsort(intensities)[::-1]
        ms2_tasks = []
        for i in idx:
            mz = mzs[i]
            intensity = intensities[i]

            # stopping criteria is after we've fragmented N ions or we found ion < min_intensity
            if fragmented_count >= self.N:
                logger.debug('Time %f Top-%d ions have been selected' % (rt, self.N))
                break

            if intensity < self.min_ms1_intensity:
                logger.debug(
                    'Time %f Minimum intensity threshold %f reached at %f, %d' % (
                        rt, self.min_ms1_intensity, intensity, fragmented_count))
                break

            # skip ion in the dynamic exclusion list of the mass spec
            if self.exclusion.is_excluded(mz, rt):
                continue

            # create a new ms2 scan parameter to be sent to the mass spec
            precursor_scan_id = scan_to_process.scan_id
            dda_scan_params = controller.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
                                                             self.mz_tol, self.rt_tol)
            new_tasks.append(dda_scan_params)
            ms2_tasks.append(dda_scan_params)
            fragmented_count += 1
            current_task_id += 1

        # add ms1 at the end
        ms1_scan_params = controller.get_ms1_scan_params()
        current_task_id += 1
        next_processed_scan_id = current_task_id
        new_tasks.append(ms1_scan_params)

        # create new exclusion items based on the scheduled ms2 tasks
        self.exclusion.update(scan_to_process, ms2_tasks)
        return new_tasks, current_task_id, next_processed_scan_id

    def _get_mzs_rt_intensities(self, scan_to_process):
        mzs = scan_to_process.mzs
        intensities = scan_to_process.intensities
        rt = scan_to_process.rt
        assert mzs.shape == intensities.shape
        return mzs, rt, intensities


class ReinforceAgent(TopNDEWAgent):
    def __init__(self, ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity, pi, min_mz, max_mz):
        super().__init__(ionisation_mode, N, isolation_width, mz_tol, rt_tol, min_ms1_intensity)
        self.pi = pi
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.seen_chems = {}

    def act(self, scan_to_process):
        # sample one action
        mzs, rt, intensities = self._get_mzs_rt_intensities(scan_to_process)
        state = self._get_state(mzs, rt, intensities)
        action = self.pi.act(state, scan_to_process.scan_id)

        # assume 10 states
        # if 0 <= action < 5:
        #     self.N = (action + 1) * 2  # N = {2, 4, 6, 8, 10}
        # elif 5 <= action < 10:
        #     self.rt_tol = (action - 5 + 1) * 5  # rt_tol = {5, 10, 15, 20, 25}

        self.N = (action + 1) * 2  # N = {2, 4, 6, ..., 20}

    def _get_state(self, mzs, rt, intensities):
        grid = int(self.max_mz - self.min_mz)
        bins = np.zeros(grid)
        for mz, intensity in zip(mzs, intensities):
            if self.exclusion.is_excluded(mz, rt):
                continue
            if intensity < self.min_ms1_intensity:
                continue
            pos = int(np.floor(mz - self.min_mz))
            try:
                bins[pos] += intensity
            except IndexError:
                bins[-1] += intensity
        bins = np.log(bins + 1)

        features = np.zeros(4)
        features[0] = np.sum(intensities > self.min_ms1_intensity)
        features[1] = len(intensities) - features[0]
        features[2] = np.log(min(intensities))
        features[3] = np.log(max(intensities))
        if np.isnan(features[2]): features[2] = 0
        state = np.concatenate((bins, features), axis=0)
        return state

    def update(self, last_scan, controller):
        super().update(last_scan, controller)

        if last_scan.ms_level >= 2:
            # update pi
            event = last_scan.fragevent
            if event is not None:  # fragmenting chems
                frag_intensity = event.parents_intensity[0]
                chem = event.chem
                if chem in self.seen_chems:
                    self.seen_chems[chem] += 1
                else:
                    self.seen_chems[chem] = 1

                reward = 0
                if frag_intensity is not None:
                    reward = np.log(frag_intensity) * 1.0 / self.seen_chems[chem]

                parent_scan_id = event.precursor_mz[0].precursor_scan_id
                assert controller.last_ms1_scan.scan_id == parent_scan_id

            else:  # fragmenting noise
                precursor = last_scan.scan_params.get(ScanParameters.PRECURSOR_MZ)[0]
                intensity = precursor.precursor_intensity
                reward = -np.log(intensity)
                parent_scan_id = controller.last_ms1_scan.scan_id

            if parent_scan_id not in self.pi.scan_id_rewards:
                self.pi.scan_id_rewards[parent_scan_id] = reward
            else:
                self.pi.scan_id_rewards[parent_scan_id] += reward


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim + 4
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.scan_ids = []
        self.scan_id_rewards = {}

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state, scan_id):
        x = torch.from_numpy(state.astype(np.float32))  # to tensor
        pdparam = self.forward(x)  # forward pass
        pd = Categorical(logits=pdparam)  # probability distribution
        action = pd.sample()  # pi(a|s) in action via pd
        log_prob = pd.log_prob(action)  # log_prob of pi(a|s)
        self.log_probs.append(log_prob)  # store for training
        self.scan_ids.append(scan_id)
        return action.item()


def train(pi, optimizer, gamma):
    rewards = []
    for scan_id in pi.scan_ids:
        reward = 0
        if scan_id in pi.scan_id_rewards:
            reward = pi.scan_id_rewards[scan_id]
        rewards.append(reward)
    pi.rewards = np.array(rewards)

    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)  # the returns
    future_ret = 0.0

    # compute the return efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    rets = rets - rets.mean()
    log_probs = torch.stack(pi.log_probs)

    # print(log_probs)
    # print(rets)
    loss = - log_probs * rets  # gradient term: Negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()  # backpropogate, compute gradients
    optimizer.step()  # gradient-ascent, update the weights
    return loss


class RandomAgent(TopNDEWAgent):
    def __init__(self, ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity, num_states):
        super().__init__(ionisation_mode, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)
        self.num_states = num_states

    def act(self, scan_to_process):
        # sample one action randomly
        action = randrange(0, self.num_states)

        # assume 10 states
        # if 0 <= action < 5:
        #     self.N = (action + 1) * 2  # N = {2, 4, 6, 8, 10}
        # elif 5 <= action < 10:
        #     self.rt_tol = (action - 5 + 1) * 5  # rt_tol = {5, 10, 15, 20, 25}

        self.N = (action + 1) * 2  # N = {2, 4, 6, ..., 20}

    def update(self, last_scan, controller):
        super().update(last_scan, controller)
