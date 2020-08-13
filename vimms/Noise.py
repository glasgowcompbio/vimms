import numpy as np


class NoPeakNoise(object):
    def get_mz(self, mz, rt, intensity, ms_level):
        return mz

    def get_intensity(self, mz, rt, intensity, ms_level):
        return intensity


class GaussianPeakNoise(NoPeakNoise):
    def __init__(self, sigma_mz, sigma_int):
        self.sigma_mz = sigma_mz
        self.sigma_int = sigma_int

    def get_mz(self, mz, rt, intensity, ms_level):
        sampled = np.random.normal(mz, self.sigma_mz, 1)  # should be specified in ppm?
        return sampled[0]

    def get_intensity(self, mz, rt, intensity, ms_level):
        sampled = np.random.normal(intensity, self.sigma_int, 1)  # maybe use the log intensity?
        return sampled[0]
