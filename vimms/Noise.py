import numpy as np


# Ensures that generators never return negative mz or intensity
def trunc_normal(mean, sigma, log_space):
    s = -1
    if not log_space:
        while s < 0:
            s = np.random.normal(mean,sigma,1)[0]
        return s
    else:
        s = np.random.normal(np.log(mean),sigma,1)[0]
        return np.exp(s)

class NoPeakNoise(object):
    def get(self, original, ms_level):
        return original


class GaussianPeakNoise(NoPeakNoise):
    def __init__(self, sigma, log_space=False):
        self.sigma = sigma
        self.log_space = log_space
    def get(self, original, ms_level):
        return trunc_normal(original,self.sigma, self.log_space)

# pass dictionary key: level, value: sigma.
# ms_levels not in the dict will not have noise added
# allows noise to be added to oa single level, or
# to all levels with different sigma
class GaussianPeakNoiseLevelSpecific(NoPeakNoise):
    def __init__(self, sigma_level_dict, log_space=False):
        self.log_space = log_space
        self.sigma_level_dict = sigma_level_dict
        
    def get(self, original, ms_level):
        if ms_level in self.sigma_level_dict:
            return trunc_normal(original, self.sigma_level_dict[ms_level], self.log_space)
        else:
            return original

def uniform_list(N,min_val,max_val):
    return list(np.random.rand(N) * (max_val - min_val) + min_val)

class UniformSpikeNoise(object):
    def __init__(self, density, max_val, min_val=0):
        self.density =  density # number of spike peaks per mz unit
        self.max_val = max_val
        self.min_val = min_val
    def sample(self, min_measurement_mz, max_measurement_mz):
        mz_range = max_measurement_mz - min_measurement_mz
        n_points = int(mz_range * self.density)
        mz_vals = uniform_list(n_points, min_measurement_mz, max_measurement_mz)
        intensity_vals = uniform_list(n_points, self.min_val, self.max_val)
        return mz_vals, intensity_vals