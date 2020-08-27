"""
Sampling classes for ChemicalMixtureCreator
"""

import numpy as np
from loguru import logger
from mass_spec_utils.library_matching.gnps import load_mgf

from vimms.Chromatograms import FunctionalChromatogram
from vimms.Common import Formula, DummyFormula, uniform_list


###############################################################################################################
# Formula samplers
###############################################################################################################

class FormulaSampler(object):
    """
    Base class for formula sampler
    """

    def sample(self, n_formulas, min_mz=50, max_mz=1000):
        raise NotImplementedError


class DatabaseFormulaSampler(FormulaSampler):
    """
    A sampler to draw formula from a database
    """

    def __init__(self, database):
        """
        Initiliases database formula sampler
        :param database: a list of Formula objects containing chemical formulae from e.g. HMDB
        """
        self.database = database

    def sample(self, n_formulas, min_mz=50, max_mz=1000):
        """
        Samples n_formulas from the specified database
        :param n_formulas: the number of formula to draw
        :param min_mz: minimum m/z of formula
        :param max_mz: maximum m/z of formula
        :return: a list of Formula objects
        """
        # filter database formulae to be within mz_range
        offset = 20  # to ensure that we have room for at least M+H
        formulas = list(set([x.chemical_formula for x in self.database]))
        sub_formulas = list(
            filter(lambda x: Formula(x).mass >= min_mz and Formula(x).mass <= max_mz - offset, formulas))
        logger.debug('{} unique formulas in filtered database'.format(len(sub_formulas)))
        chosen_formulas = np.random.choice(sub_formulas, size=n_formulas, replace=False)
        logger.debug('Sampled formulas')
        return [Formula(f) for f in chosen_formulas]


class UniformMZFormulaSampler(FormulaSampler):
    """
    A sampler to generate formula uniformly between min_mz to max_mz, so just mz rather then formulas.
    Resulting in UnknownChemical objects instead of known_chemical ones.
    """

    def sample(self, n_formulas, min_mz=50, max_mz=1000):
        """
        Samples n_formulas uniformly between min_mz and max_mz
        :param n_formulas: the number of formula to draw
        :param min_mz: minimum m/z of formula
        :param max_mz: maximum m/z of formula
        :return: a list of Formula objects
        """
        mz_list = np.random.rand(n_formulas) * (max_mz - min_mz) + min_mz
        return [DummyFormula(m) for m in mz_list]


class PickEverythingFormulaSampler(DatabaseFormulaSampler):
    """
    A sampler that doesn't do anything and just return everything in the database
    """

    def __init__(self, database):
        """
        Initiliases database formula sampler
        :param database: a list of Formula objects containing chemical formulae from e.g. HMDB
        """
        self.database = database

    def sample(self, n_formulas):
        """
        Just return everything from the database
        :param n_formulas: ignored?
        :return: all formulae from the database
        """
        return list(self.database)


###############################################################################################################
# Samplers for RT and intensity when initialising a Formula
###############################################################################################################


class RTAndIntensitySampler(object):
    """
    Base class for RT and intensity sampler. Usually used when initialising a formula object.
    """

    def sample(self, formula):
        raise NotImplementedError


class UniformRTAndIntensitySampler(RTAndIntensitySampler):
    """
    A sampler to sample RT and log intensity uniformly.
    See class def for min and max log intensity.
    Returns actual intensity, but samples in log space.
    """

    def __init__(self, min_rt=0, max_rt=1600, min_log_intensity=np.log(1e4), max_log_intensity=np.log(1e7)):
        """
        Initialises uniform RT and intensity sampler
        :param min_rt: minimum RT
        :param max_rt: maximum RT
        :param min_log_intensity: minimum log intensity
        :param max_log_intensity: maximum log intensity
        """
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity

    def sample(self, formula):
        """
        Samples RT and log intensity uniformly between (min_rt, max_rt) and (min_log_intensity, max_log_intensity)
        :param formula: the formula to condition on (can be ignored)
        :return: a tuple of (RT, intensity)
        """
        rt = np.random.rand() * (self.max_rt - self.min_rt) + self.min_rt
        log_intensity = np.random.rand() * (self.max_log_intensity - self.min_log_intensity) + self.min_log_intensity
        return rt, np.exp(log_intensity)


###############################################################################################################
# Chromatogram samplers
###############################################################################################################


class ChromatogramSampler(object):
    """
    Base class for chromatogram sampler.
    """

    def sample(self, formula, rt, intensity):
        raise NotImplementedError


class GaussianChromatogramSampler(ChromatogramSampler):
    """
    A sampler to return Gaussian-shaped chromatogram
    """

    def __init__(self, sigma=10):
        self.sigma = sigma

    def sample(self, formula, rt, intensity):
        """
        Sample a Gaussian-shaped chromatogram
        :param formula: the formula to condition on (can be ignored)
        :param rt: RT to condition on (can be ignored)
        :param intensity: intensity to condition on (can be ignored)
        :return:
        """
        return FunctionalChromatogram('normal', [0, self.sigma])


###############################################################################################################
# MS2 samplers
###############################################################################################################


class MS2Sampler(object):
    """
    Base class for MS2 sampler
    """

    def sample(self, formula):
        raise NotImplementedError


class UniformMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 peaks uniformly between min_mz and the mass of the formula.
    """

    def __init__(self, poiss_peak_mean=10, min_mz=50, min_proportion=0.1, max_proportion=0.8):
        """
        Initialises uniform MS2 sampler
        :param poiss_peak_mean: the mean of the Poisson distribution used to draw the number of peaks
        :param min_mz: minimum m/z
        :param min_proportion: minimum proportion from the parent MS1 peak intensities
        :param max_proportion: maximum proportion from the parent MS1 peak intensities
        """
        self.poiss_peak_mean = poiss_peak_mean
        self.min_mz = min_mz
        self.min_proportion = min_proportion  # proportion of parent intensity shared by MS2
        self.max_proportion = max_proportion

    def sample(self, formula):
        """
        Samples n_peaks of MS2 peaks uniformly between min_mz and the exact mass of the formula.
        The intensity is also randomly sampled between between min_proportion and max_proportion of the parent
        formula intensity
        :param formula: the parent formula
        :return: a tuple of (mz_list, intensity_list, parent_proportion)
        """
        n_peaks = np.random.poisson(self.poiss_peak_mean)
        max_mz = formula.compute_exact_mass()
        mz_list = uniform_list(n_peaks, self.min_mz, max_mz)
        intensity_list = uniform_list(n_peaks, 0, 1)

        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                            self.min_proportion

        return mz_list, intensity_list, parent_proportion


class CRPMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 peaks following the CRP.
    """

    def __init__(self, n_draws=1000, min_mz=50, min_proportion=0.1, max_proportion=0.8, alpha=1, base='uniform'):
        self.n_draws = n_draws
        self.min_mz = min_mz
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.alpha = alpha
        self.base = base

    def sample(self, formula):
        assert self.base == 'uniform'
        max_mz = formula.compute_exact_mass()
        unique_vals = [self._base_sample(max_mz)]
        counts = [1]
        for i in range(self.n_draws - 1):
            temp = counts + [self.alpha]
            s = sum(temp)
            probs = [t / s for t in temp]
            choice = np.random.choice(len(temp), p=probs)
            if choice == len(unique_vals):
                # new value
                unique_vals.append(self._base_sample(max_mz))
                counts.append(1)
            else:
                counts[choice] += 1

        mz_list = unique_vals
        s = sum(counts)
        intensity_list = [c / s for c in counts]
        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                            self.min_proportion

        return mz_list, intensity_list, parent_proportion

    def _base_sample(self, max_mz):
        return np.random.rand() * (max_mz - self.min_mz) + self.min_mz


class MGFMS2Sampler(MS2Sampler):
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8, max_peaks=0, replace=False):
        self.mgf_file = mgf_file
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.replace = replace  # sample with replacement

        # load the mgf
        spectra_dict = load_mgf(self.mgf_file)

        # turn into a list where the last item is the number of times this one has been sampled
        self.spectra_list = [[s.precursor_mz, s, 0] for s in spectra_dict.values()]

        # filter to remove those with more than  max_peaks (if max_peaks > 0)
        if max_peaks > 0:
            self.spectra_list = list(filter(lambda x: len(x[1].peaks) <= max_peaks, self.spectra_list))

        # sort by precursor mz
        self.spectra_list.sort(key=lambda x: x[0])
        logger.debug("Loaded {} spectra from {}".format(len(self.spectra_list), self.mgf_file))

    def sample(self, formula):
        formula_mz = formula.mass
        sub_spec = list(filter(lambda x: x[0] < formula_mz, self.spectra_list))
        if len(sub_spec) == 0:
            sub_spec = self.spectra_list  # if there aren't any smaller than the mz, we just take any one

        # sample one. If replace == True we take any, if not we only take those that have not been sampled before
        found_permissable = False
        n_attempts = 0
        while not found_permissable:
            n_attempts += 1
            spec = np.random.choice(len(sub_spec))
            if self.replace == True or sub_spec[spec][2] == 0 or n_attempts > 100:
                found_permissable = True

        sub_spec[spec][2] += 1  # add one to the count
        spectrum = sub_spec[spec][1]
        mz_list, intensity_list = zip(*spectrum.peaks)
        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                            self.min_proportion

        return mz_list, intensity_list, parent_proportion


class ExactMatchMS2Sampler(MS2Sampler):
    # to be completed. Where we have particular formulas and we
    # have a particular spectrum for each exact formula...
    def __init__(self):
        pass
