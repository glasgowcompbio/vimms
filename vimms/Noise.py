import numpy as np

from vimms.Common import uniform_list


def trunc_normal(mean, sigma, log_space):
    """
    Ensures that generators never return negative mz or intensity
    :param mean: mean of gaussian distribution to sample from
    :param sigma: variance of gaussian distribution to sample from
    :param log_space: whether to sample in log space
    :return: the sampled value
    """
    s = -1
    if not log_space:
        while s < 0:
            s = np.random.normal(mean, sigma, 1)[0]
        return s
    else:
        s = np.random.normal(np.log(mean), sigma, 1)[0]
        return np.exp(s)


class NoPeakNoise(object):
    """
    The base peak noise object that doesn't add any noise
    """
    def get(self, original, ms_level):
        """
        Get the original value back. No noise if applied.
        :param original: The original value
        :param ms_level: The ms level
        :return: the original value
        """
        return original


class GaussianPeakNoise(NoPeakNoise):
    """
    Adds Gaussian noise to peaks
    """
    def __init__(self, sigma, log_space=False):
        """
        Initialises Gaussian peak noise
        :param sigma: the variance
        :param log_space: whether to sample in log space
        """
        self.sigma = sigma
        self.log_space = log_space

    def get(self, original, ms_level):
        """
        Gets peak measurement with gaussian noise applied
        :param original:
        :param ms_level:
        :return:
        """
        return trunc_normal(original, self.sigma, self.log_space)


class GaussianPeakNoiseLevelSpecific(NoPeakNoise):
    """
    Adds ms-level specific Gaussian noise to peaks
    Pass dictionary key: level, value: sigma.
    ms_levels not in the dict will not have noise added
    allows noise to be added to oa single level, or
    to all levels with different sigma
    """
    def __init__(self, sigma_level_dict, log_space=False):
        self.log_space = log_space
        self.sigma_level_dict = sigma_level_dict

    def get(self, original, ms_level):
        if ms_level in self.sigma_level_dict:
            return trunc_normal(original, self.sigma_level_dict[ms_level], self.log_space)
        else:
            return original


class UniformSpikeNoise(object):
    def __init__(self, density, max_val, min_val=0):
        self.density = density  # number of spike peaks per mz unit
        self.max_val = max_val
        self.min_val = min_val

    def sample(self, min_measurement_mz, max_measurement_mz):
        mz_range = max_measurement_mz - min_measurement_mz
        n_points = int(mz_range * self.density)
        mz_vals = uniform_list(n_points, min_measurement_mz, max_measurement_mz)
        intensity_vals = uniform_list(n_points, self.min_val, self.max_val)
        return mz_vals, intensity_vals
