# Sampling classes for ChemicalMixtureCreator
import numpy as np
import bisect
from loguru import logger

from mass_spec_utils.library_matching.gnps import load_mgf

from vimms.Chromatograms import FunctionalChromatogram
from vimms.Common import Formula, DummyFormula

class FormulaSampler(object):
    def sample(self,n_formulas,min_mz=50,max_mz=1000):
        raise NotImplementedError

class DatabaseFormulaSampler(FormulaSampler):
    def __init__(self, database):
        self.database = database

    def sample(self,n_formulas,min_mz=50,max_mz=1000):
        # filter HMDB to witin mz_range
        offset = 20 # to ensure that we have room for at least M+H
        formulas = list(set([x.chemical_formula for x in self.database]))
        sub_formulas = list(filter(lambda x: Formula(x).mass  >= min_mz and Formula(x).mass <= max_mz - offset,formulas))
        logger.debug('{} unique formulas in filtered database'.format(len(sub_formulas)))
        chosen_formulas = np.random.choice(sub_formulas, size=n_formulas, replace=False)
        logger.debug('Sampled formulas')
        return [Formula(f) for f in chosen_formulas]

class UniformMZFormulaSampler(FormulaSampler):
    def sample(self,n_formulas,min_mz=50,max_mz=1000):
        offset = 20
        mz_list = np.random.rand(n_formulas) * (max_mz - min_mz) + min_mz
        return [DummyFormula(m) for m in  mz_list]


class RTAndIntensitySampler(object):
    def sample(self,formula):
        raise NotImplementedError

class UniformRTAndIntensitySampler(RTAndIntensitySampler):
    def __init__(self,min_rt = 0, max_rt = 1600, min_log_intensity = np.log(1e4), max_log_intensity = np.log(1e7)):
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity
    def sample(self,formula): 
        rt = np.random.rand() * (self.max_rt - self.min_rt) + self.min_rt
        log_intensity = np.random.rand() * (self.max_log_intensity - self.min_log_intensity) + self.min_log_intensity
        return rt, np.exp(log_intensity)

class ChromatogramSampler(object):
    def sample(self,formula,rt,intensity):
        raise NotImplementedError

class GaussianChromatogramSampler(ChromatogramSampler):
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


class MGFMS2Sampler(MS2Sampler):
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8, max_peaks=0, replace=False):
        self.mgf_file = mgf_file
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.replace = replace # sample with replacement

        # load the mgf
        spectra_dict = load_mgf(self.mgf_file)
        
        # turn into a list where the last item is the number of times this one has been sampled
        self.spectra_list = [[s.precursor_mz,s,0] for s in spectra_dict.values()]

        # filter to remove those with more than  max_peaks (if max_peaks > 0)
        if max_peaks > 0:
            self.spectra_list = list(filter(lambda x: len(x[1].peaks) <= max_peaks, self.spectra_list))
        
        # sort by precursor mz
        self.spectra_list.sort(key = lambda x: x[0])
        logger.debug("Loaded {} spectra from {}".format(len(self.spectra_list),self.mgf_file))
    
    def sample(self, formula):
        formula_mz = formula.mass
        sub_spec = list(filter(lambda x: x[0] < formula_mz,self.spectra_list))
        if len(sub_spec) == 0:
            sub_spec = self.spectra_list # if there aren't any smaller than the mz, we just take any one

        # sample one. If replace == True we take any, if not we only those that have not been sampled before
        found_permissable = False
        n_attempts = 0
        while not found_permissable:
            n_attempts += 1
            spec = np.random.choice(len(sub_spec))
            if self.replace == True or sub_spec[spec][2] == 0 or n_attempts > 100:
                found_permissable = True

        sub_spec[spec][2] += 1 # add one to the count
        spectrum = sub_spec[spec][1]
        mz_list,intensity_list = zip(*spectrum.peaks)
        s = sum(intensity_list)
        intensity_list = [i/s for i in intensity_list]
        
        parent_proportion = np.random.rand()*(self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion

class ExactMatchMS2Sampler(MS2Sampler):
    # to be completed. Where we have particular formulas and we
    # have a particular spectrum for each exact formula...

