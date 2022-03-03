"""
Models chromatographic column in an LC-MS process.
This is particularly useful to investigate the effect of retention time drifts on chemicals.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np


class Column():
    """
    Defines a base Column class that operates on a dataset and having some noise parameter
    """
    def __init__(self, dataset, noise_sd):
        """
        Create a column object

        Args:
            dataset: the set of Chemicals that passes through this column
            noise_sd: the noise standard deviation sampled from a normal distribution
        """
        self.dataset = dataset
        self.dataset_rts = np.array([chem.rt for chem in self.dataset])
        self.dataset_apex_rts = np.array(
            [chem.get_apex_rt() for chem in self.dataset])
        self.noise_sd = noise_sd
        self.offsets, self.true_drift_function = self._get_offsets()

    def _get_offsets(self):
        """
        Computes the offset and the offset function

        Returns: the offset and the offset function

        """
        true_offset_function = np.array([0.0 for chem in self.dataset])
        offsets = true_offset_function + np.random.normal(0, self.noise_sd,
                                                          len(self.dataset))
        return offsets, true_offset_function

    def get_dataset(self):
        """
        Gets a modified dataset with column (RT) noise applied
        Returns: a new list of Chemicals where its noise have been modified by the column

        """
        new_dataset = []
        for i, chem in enumerate(self.dataset):
            new_chem = copy.deepcopy(chem)
            new_chem.rt += self.offsets[i]
            new_dataset.append(new_chem)
        return new_dataset

    def get_chemical(self, idx):
        """
        Gets the chemical at the specified index (?)
        Args:
            idx: the index to search

        Returns: the chemical and its offset (?)

        """
        return self.dataset[idx] + self.offsets[idx]

    def plot_drift(self):
        """
        Plot the drift

        Returns: None

        """
        order = np.argsort(self.dataset_rts)
        plt.figure(figsize=(12, 8))
        plt.plot(self.dataset_rts[order], self.true_drift_function[order], 'b')
        plt.plot(self.dataset_rts[order],
                 self.true_drift_function[order] + 1.95 * self.noise_sd, 'b--')
        plt.plot(self.dataset_rts[order],
                 self.true_drift_function[order] - 1.95 * self.noise_sd, 'b--')
        plt.plot(self.dataset_rts, self.offsets, 'ro')
        plt.ylabel('Drift Amount')
        plt.xlabel('Base RT')
        plt.show()

    def plot_drift_distribution(self):
        """
        Plot drift distribution

        Returns: None

        """
        order = np.argsort(self.dataset_rts)
        plt.figure(figsize=(12, 8))
        for i in range(100):
            offsets, true_drift_function = self._get_offsets()
            plt.plot(self.dataset_rts[order], true_drift_function[order])
        plt.ylabel('Drift Amount')
        plt.xlabel('Base RT')
        plt.show()


class CleanColumn(Column):
    """
    A clean column with no RT noise
    """
    def __init__(self, dataset):
        """
        Create a clean column object
        Args:
            dataset: the set of Chemicals that passes through this column
        """
        super().__init__(dataset, 0.0)


class LinearColumn(Column):
    """
    A column with linear drift in the RT
    """
    def __init__(self, dataset, noise_sd, intercept_params, linear_params):
        """
        Create a linear drift column
        Args:
            dataset: the set of Chemicals that passes through this column
            noise_sd: noise standard deviation
            intercept_params: intercept parameters
            linear_params: linear parameters
        """
        self.intercept_params = intercept_params
        self.linear_params = linear_params
        self.intercept_term = np.random.normal(self.intercept_params[0],
                                               self.intercept_params[1])
        self.linear_term = np.random.normal(self.linear_params[0],
                                            self.linear_params[1])
        super().__init__(dataset, noise_sd)

    @staticmethod
    def from_fixed_offsets(dataset, noise_sd, intercept_term, linear_term):
        """
        From fixed offsets

        Args:
            dataset:
            noise_sd:
            intercept_term:
            linear_term:

        Returns: ???

        """
        new = LinearColumn(dataset, noise_sd, (0, 0), (0, 0))
        new.intercept_term, new.linear_term = intercept_term, linear_term
        new.offsets, new.true_drift_function = new._get_offsets()
        return new

    def _get_offsets(self):
        """
        Get offsets

        Returns: ???

        """
        true_offset_function = self.intercept_term + \
            self.linear_term * self.dataset_apex_rts
        offsets = true_offset_function + np.random.normal(0, self.noise_sd,
                                                          len(self.dataset))
        return offsets, true_offset_function

    def drift_fn(self, roi, injection_number):
        """
        Drift function

        Args:
            roi:
            injection_number:

        Returns: ???

        """
        """f(rt) = rt + (m * rt + c)
        rt + m * rt = f(rt) - c
        rt(1 + m) = f(rt) - c
        rt = (f(rt) - c) / (1 + m)"""
        rt = roi.estimate_apex()
        return rt - (rt - self.intercept_term) / (
            1 + self.linear_term), {}  # this doesn't account for noise?


class GaussianProcessColumn(Column):
    """
    A gaussian-process based column
    """
    def __init__(self, dataset, noise_sd, rbf_params, intercept_params,
                 linear_params):
        """
        Create a gaussian process drift column
        Args:
            dataset:
            noise_sd:
            rbf_params:
            intercept_params:
            linear_params:
        """
        self.rbf_params = rbf_params
        self.intercept_params = intercept_params
        self.linear_params = linear_params
        super().__init__(dataset, noise_sd)

    def _get_offsets(self):
        """
        Get offset

        Returns: ???

        """
        intercept_term = np.random.normal(self.intercept_params[0],
                                          self.intercept_params[1])
        linear_term = np.random.normal(self.linear_params[0],
                                       self.linear_params[1])
        mean = intercept_term + linear_term * self.dataset_apex_rts
        return self._draw_offset(mean)

    def _draw_offset(self, mean):
        """
        Draw offset

        Args:
            mean:

        Returns: ???

        """
        N = len(self.dataset_apex_rts)
        K = np.zeros((N, N), np.double)
        for n in range(N):
            for m in range(N):
                K[n, m] = self.rbf_params[0] * np.exp(
                    -(1. / self.rbf_params[1]) * (self.dataset_apex_rts[n] -
                                                  self.dataset_apex_rts[
                                                      m]) ** 2)
        true_offset_function = np.random.multivariate_normal(mean, K)
        offsets = true_offset_function + np.random.normal(0, self.noise_sd, N)
        return offsets, true_offset_function
