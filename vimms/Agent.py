# Agent.py
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from vimms.Common import ScanParameters
from vimms.Exclusion import BoxHolder, ExclusionItem


class AbstractAgent(object):
    def __init__(self):
        self.task_list = []

    def next_tasks(self, scan, controller, current_task_id):
        raise NotImplementedError

    def update(self, last_scan, controller):
        raise NotImplementedError


class FullScanAgent(AbstractAgent):
    def __init__(self):
        super().__init__()

    def next_tasks(self, scan, controller, current_task_id):
        new_tasks = []
        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        current_task_id += 1
        next_processed_scan_id = current_task_id
        self.task_list.append((controller, ms1_scan_params))
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass


class TopNAgent(AbstractAgent):
    def __init__(self, N, min_ms1_intensity):
        super().__init__()
        self.N = N
        self.min_ms1_intensity = min_ms1_intensity
        self.mz_tol = 10
        self.rt_tol = 15
        self.isolation_width = 0.7
        self.exclusion = BoxHolder()

    def next_tasks(self, scan, controller, current_task_id):
        new_tasks = []
        mzs = scan.mzs
        intensities = scan.intensities
        assert mzs.shape == intensities.shape
        rt = scan.rt
        idx = np.argsort(intensities)[::-1]

        fragmented_count = 0
        for i in idx:
            if fragmented_count >= self.N:
                break
            mz = mzs[i]
            intensity = intensities[i]

            if intensity < self.min_ms1_intensity:
                break

            if self.exclusion.is_in_box(mz, rt):  # will always return false in this controller, but used in children
                continue

            precursor_scan_id = scan.scan_id
            dda_scan_params = controller.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
                                                             self.mz_tol, self.rt_tol)
            new_tasks.append(dda_scan_params)
            self.task_list.append((controller, dda_scan_params))
            fragmented_count += 1
            current_task_id += 1

        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        self.task_list.append((controller, ms1_scan_params))
        current_task_id += 1
        next_processed_scan_id = current_task_id

        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        pass


class TopNDEWAgent(TopNAgent):
    def __init__(self, N, min_ms1_intensity, mz_tol, rt_tol):
        super().__init__(N, min_ms1_intensity)
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol

    def update(self, last_scan, controller):
        rt = last_scan.rt
        if last_scan.ms_level >= 2:  # if ms-level is 2, it's a custom scan and we should always know its scan parameters
            assert last_scan.scan_params is not None
            for precursor in last_scan.scan_params.get(ScanParameters.PRECURSOR_MZ):
                # add dynamic exclusion item to the exclusion list to prevent the same precursor ion being fragmented
                # multiple times in the same mz and rt window
                # Note: at this point, fragmentation has occurred and time has been incremented! so the time when
                # items are checked for dynamic exclusion is the time when MS2 fragmentation occurs
                # TODO: we need to add a repeat count too, i.e. how many times we've seen a fragment peak before
                #  it gets excluded (now it's basically 1)

                # TODO: check if already excluded and, if so, just move the time
                mz = precursor.precursor_mz
                mz_lower = mz * (1 - self.mz_tol / 1e6)
                mz_upper = mz * (1 + self.mz_tol / 1e6)
                rt_lower = rt - self.rt_tol
                rt_upper = rt + self.rt_tol
                x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper,
                                  frag_at=rt)
                self.exclusion.add_box(x)


class Pi(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim + 4
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(self.in_dim, self.hidden_dim),
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


class ReinforceAgent(AbstractAgent):
    def __init__(self, pi, N, min_ms1_intensity, mz_tol, rt_tol, min_mz, max_mz):
        super().__init__()
        self.pi = pi
        self.N = N
        self.min_ms1_intensity = min_ms1_intensity
        self.exclusion = BoxHolder()
        self.mz_tol = mz_tol
        self.rt_tol = rt_tol
        self.isolation_width = 0.7
        self.min_mz = min_mz
        self.max_mz = max_mz

    def _get_state(self, mzs, rt, intensities):
        grid = int(self.max_mz - self.min_mz)
        bins = np.zeros(grid)
        for mz, intensity in zip(mzs, intensities):
            if self.exclusion.is_in_box(mz, rt):
                continue
            if intensity < self.min_ms1_intensity:
                continue
            pos = int(np.floor(mz - self.min_mz))
            try:
                bins[pos] += np.log(intensity)
            except IndexError:
                bins[-1] += np.log(intensity)

        features = np.zeros(4)
        features[0] = np.sum(intensities > self.min_ms1_intensity)
        features[1] = len(intensities) - features[0]
        features[2] = np.log(min(intensities))
        features[3] = np.log(max(intensities))
        if np.isnan(features[2]): features[2] = 0
        state = np.concatenate((bins, features), axis=0)
        return state

    def next_tasks(self, scan, controller, current_task_id):
        mzs = scan.mzs
        intensities = scan.intensities
        rt = scan.rt
        assert mzs.shape == intensities.shape

        # sample one action
        state = self._get_state(mzs, rt, intensities)
        action = self.pi.act(state, scan.scan_id)
        if 0 <= action < 5:
            self.N = (action+1)*2
        elif 5 <= action < 10:
            self.rt_tol = (action-5+1) * 5

        # generate scan commands according to action
        new_tasks = []
        idx = np.argsort(intensities)[::-1]
        fragmented_count = 0
        for i in idx:
            if fragmented_count >= self.N:
                break

            mz = mzs[i]
            intensity = intensities[i]

            if intensity < self.min_ms1_intensity:
                break

            if self.exclusion.is_in_box(mz, rt):
                continue

            precursor_scan_id = scan.scan_id
            dda_scan_params = controller.get_ms2_scan_params(mz, intensity, precursor_scan_id, self.isolation_width,
                                                             self.mz_tol, self.rt_tol)
            new_tasks.append(dda_scan_params)
            self.task_list.append((controller, dda_scan_params))
            fragmented_count += 1
            current_task_id += 1

        ms1_scan_params = controller.get_ms1_scan_params()
        new_tasks.append(ms1_scan_params)
        self.task_list.append((controller, ms1_scan_params))
        current_task_id += 1
        next_processed_scan_id = current_task_id
        return new_tasks, current_task_id, next_processed_scan_id

    def update(self, last_scan, controller):
        rt = last_scan.rt
        if last_scan.ms_level >= 2:

            # update exclusion boxes
            assert last_scan.scan_params is not None
            for precursor in last_scan.scan_params.get(ScanParameters.PRECURSOR_MZ):
                mz = precursor.precursor_mz
                mz_lower = mz * (1 - self.mz_tol / 1e6)
                mz_upper = mz * (1 + self.mz_tol / 1e6)
                rt_lower = rt - self.rt_tol
                rt_upper = rt + self.rt_tol
                x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper,
                                  frag_at=rt)
                self.exclusion.add_box(x)

            # update pi
            event = last_scan.fragevent
            if event is not None: # fragmenting chems
                frag_intensity = event.parents_intensity[0]
                reward = 0
                if frag_intensity is not None:
                    reward = np.log(frag_intensity)
                # reward = +1

                parent_scan_id = event.precursor_mz[0].precursor_scan_id
                assert controller.last_ms1_scan.scan_id == parent_scan_id
                if parent_scan_id not in self.pi.scan_id_rewards:
                    self.pi.scan_id_rewards[parent_scan_id] = reward
                else:
                    self.pi.scan_id_rewards[parent_scan_id] += reward

            else: # fragmenting noise
                parent_scan_id = controller.last_ms1_scan.scan_id
                reward = -10
                if parent_scan_id not in self.pi.scan_id_rewards:
                    self.pi.scan_id_rewards[parent_scan_id] = reward
                else:
                    self.pi.scan_id_rewards[parent_scan_id] += reward
