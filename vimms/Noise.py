import numpy as np

from vimms.Common import uniform_list


def trunc_normal(mean, sigma, log_space):
    """
    Ensures that generators never return negative mz or intensity

    Args:
        mean: mean of gaussian distribution to sample from
        sigma: variance of gaussian distribution to sample from
        log_space: whether to sample in log space

    Returns: the sampled value

    """
    s = -1
    if not log_space:
        while s < 0:
            s = np.random.normal(mean, sigma, 1)[0]
        return s
    else:
        s = np.random.normal(np.log(mean), sigma, 1)[0]
        return np.exp(s)


class NoPeakNoise():
    """
    The base peak noise object that doesn't add any noise
    """

    def get(self, original, ms_level):
        """
        Get the original value back. No noise if applied.

        Args:
            original: The original value
            ms_level: The ms level

        Returns: the original value (unused)

        """
        return original


class GaussianPeakNoise(NoPeakNoise):
    """
    Adds Gaussian noise to peaks
    """

    def __init__(self, sigma, log_space=False):
        """
        Initialises Gaussian peak noise

        Args:
            sigma: the variance
            log_space: whether to sample in log space
        """
        self.sigma = sigma
        self.log_space = log_space

    def get(self, original, ms_level):
        """
        Get peak measurement with gaussian noise applied

        Args:
            original: original value
            ms_level: ms level

        Returns: peak measurement with gaussian noise applied

        """
        return trunc_normal(original, self.sigma, self.log_space)


class GaussianPeakNoiseLevelSpecific(NoPeakNoise):
    """
    Adds ms-level specific Gaussian noise to peaks
    """

    def __init__(self, sigma_level_dict, log_space=False):
        """
        Create a gaussian peak noise level specific

        Args:
            sigma_level_dict: key: level, value: sigma.
                              ms_levels not in the dict will not have noise added
                              allows noise to be added to oa single level, or
                              to all levels with different sigma
            log_space: whether to log or not
        """
        self.log_space = log_space
        self.sigma_level_dict = sigma_level_dict

    def get(self, original, ms_level):
        if ms_level in self.sigma_level_dict:
            return trunc_normal(original, self.sigma_level_dict[ms_level],
                                self.log_space)
        else:
            return original


class UniformSpikeNoise():
    """
    A class to add uniform spike noise to the data
    """
    def __init__(self, density, max_val, min_val=0, min_mz=None, max_mz=None):
        """
        Create a UniformSpikeNoise class
        Args:
            density: number of spike peaks per mz unit
            max_val: maximum value of spike
            min_val: minimum value of spike
            min_mz: maximum m/z
            max_mz: minimum m/z
        """
        self.density = density
        self.max_val = max_val
        self.min_val = min_val
        self.min_mz = min_mz
        self.max_mz = max_mz

    def sample(self, min_measurement_mz, max_measurement_mz):
        if self.min_mz is not None:
            min_measurement_mz = self.min_mz
        if self.max_mz is not None:
            max_measurement_mz = self.max_mz
        mz_range = max_measurement_mz - min_measurement_mz
        n_points = max(int(mz_range * self.density), 1)
        mz_vals = uniform_list(
            n_points, min_measurement_mz, max_measurement_mz)
        intensity_vals = uniform_list(n_points, self.min_val, self.max_val)
        return mz_vals, intensity_vals
