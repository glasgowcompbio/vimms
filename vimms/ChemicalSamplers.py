# Sampling classes for ChemicalMixtureCreator
import numpy as np

from vimms.Chromatograms import FunctionalChromatogram


class RTAndIntensitySampler(object):
    def sample(self,formula):
        raise NotImplementedError

class UniformRTAndIntensitySampler(RTAndIntensitySampler):
    def __init__(self,min_rt = 0, max_rt = 1600, min_log_intensity = np.log(1e4), max_log_intensity = np.log(1e7)):
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity
    def sample(self,formula): #maybe parameterise by mz?
        rt = np.random.rand() * (self.max_rt - self.min_rt) + self.min_rt
        log_intensity = np.random.rand() * (self.max_log_intensity - self.min_log_intensity) + self.min_log_intensity
        return rt, np.exp(log_intensity)

class ChromatogramSampler(object):
    def sample(self,formula,rt,intensity):
        raise NotImplementedError

class GaussianChromatogramSampler(ChrmoatogramSampler):
    def __init__(self, sigma = 10):
        self.sigma = sigma
    def sample(self, formula, rt, intensity):
        return FunctionalChromatogram('normal',[0,self.sigma])


class MS2Sampler(object):
    def sample(self,formula):
        raise NotImplementedError

class UniformMS2Sampler(MS2Sampler):
    def __init__(self,poiss_peak_mean=10, min_mz=50, min_proportion=0.1, max_proportion=0.8):
        self.poiss_peak_mean = poiss_peak_mean
        self.min_mz = min_mz
        self.min_proportion = min_proportion # proportion of parent intensity shared by MS2
        self.max_proportion = max_proportion
    def sample(self,formula):
        n_peaks = np.random.poisson(self.poiss_peak_mean)
        max_mz = formula.compute_exact_mass()
        mz_list = uniform_list(n_peaks, self.min_mz, max_mz)
        intensity_list = uniform_list(n_peaks, 0, 1)

        s = sum(intensity_list)
        intensity_list = [i/s for i in intensity_list]
        parent_proportion = np.random.rand()*(self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion


class CRPMS2Sampler(MS2Sampler):
    def __init__(self,n_draws=1000, min_mz=50, min_proportion=0.1, max_proportion=0.8, alpha=1, base='uniform'):
        self.n_draws = n_draws
        self.min_mz = min_mz
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.alpha = alpha
        self.base = base
    def sample(self,formula):
        assert self.base == 'uniform'
        max_mz = formula.compute_exact_mass()
        unique_vals = [self._base_sample(max_mz)]
        counts  = [1]
        for i in range(self.n_draws - 1):
            temp = counts + [self.alpha]
            s = sum(temp)
            probs = [t/s for t in temp]
            choice = np.random.choice(len(temp),p=probs)
            if choice == len(unique_vals):
                # new value
                unique_vals.append(self._base_sample(max_mz))
                counts.append(1)
            else:
                counts[choice] += 1
        
        mz_list = unique_vals
        s = sum(counts)
        intensity_list = [c/s for c in counts]

        parent_proportion = np.random.rand()*(self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion


    def _base_sample(self,max_mz):
        return np.random.rand()*(max_mz - self.min_mz) + self.min_mz

