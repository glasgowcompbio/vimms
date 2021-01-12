import copy

import numpy as np
import matplotlib.pyplot as plt


class Column(object):
    def __init__(self, dataset, noise_sd):
        self.dataset = dataset
        self.dataset_rts = np.array([chem.rt for chem in self.dataset])
        self.noise_sd = noise_sd
        self.offsets, self.true_drift_function = self._get_offsets()

    def _get_offsets(self):
        true_offset_function = np.array([0.0 for chem in self.dataset])
        offsets = true_offset_function + np.random.normal(0, self.noise_sd, len(self.dataset))
        return offsets, true_offset_function

    def get_dataset(self):
        new_dataset = []
        for chem, rt_shift in zip(self.dataset, self.offsets):
            new_chem = copy.deepcopy(chem)
            new_chem.rt += rt_shift
            new_dataset.append(new_chem)
        return new_dataset

    def get_chemical(self, idx):
        return self.dataset[idx] + self.offsets[idx]

    def plot_drift(self):
        order = np.argsort(self.dataset_rts)
        plt.figure(figsize=(12, 8))
        plt.plot(self.dataset_rts[order], self.true_drift_function[order], 'b')
        plt.plot(self.dataset_rts[order], self.true_drift_function[order] + 1.95 * self.noise_sd, 'b--')
        plt.plot(self.dataset_rts[order], self.true_drift_function[order] - 1.95 * self.noise_sd, 'b--')
        plt.plot(self.dataset_rts, self.offsets, 'ro')
        plt.show()

    def plot_drift_distribution(self):
        order = np.argsort(self.dataset_rts)
        plt.figure(figsize=(12, 8))
        for i in range(100):
            offsets, true_drift_function = self._get_offsets()
            plt.plot(self.dataset_rts[order], true_drift_function[order])
        plt.xlabel('Base RT')
        plt.show()


class CleanColumn(Column):
    def __init__(self, dataset):
        super().__init__(dataset, 0.0)


class GaussianProcessColumn(Column):
    def __init__(self, dataset, noise_sd, rbf_params, intercept_params, linear_params):
        self.rbf_params = rbf_params
        self.intercept_params = intercept_params
        self.linear_params = linear_params
        super().__init__(dataset, noise_sd)

    def _get_offsets(self):
        intercept_term = np.random.normal(self.intercept_params[0], self.intercept_params[1])
        linear_term = np.random.normal(self.linear_params[0], self.linear_params[1])
        mean = intercept_term + linear_term * self.dataset_rts
        return self._draw_offset(mean)

    def _draw_offset(self, mean):
        N = len(self.dataset_rts)
        K = np.zeros((N, N), np.double)
        for n in range(N):
            for m in range(N):
                K[n, m] = self.rbf_params[0] * np.exp(-(1. / self.rbf_params[1]) * (self.dataset_rts[n] - self.dataset_rts[m]) ** 2)
        true_offset_function = np.random.multivariate_normal(mean, K)
        offsets = true_offset_function + np.random.normal(0, self.noise_sd, N)
        return offsets, true_offset_function


