"""
Sampling classes for ChemicalMixtureCreator
"""

import numpy as np
from loguru import logger
from mass_spec_utils.library_matching.gnps import load_mgf
from mass_spec_utils.data_import.mzml import MZMLFile

from vimms.Chromatograms import FunctionalChromatogram, ConstantChromatogram, EmpiricalChromatogram
from vimms.Common import Formula, DummyFormula, uniform_list, DEFAULT_MS1_SCAN_WINDOW, DEFAULT_MSN_SCAN_WINDOW, \
                        POSITIVE, NEGATIVE, PROTON_MASS
from vimms.Roi import make_roi, RoiParams

MIN_MZ = DEFAULT_MS1_SCAN_WINDOW[0]
MAX_MZ = DEFAULT_MS1_SCAN_WINDOW[1]
MIN_MZ_MS2 = DEFAULT_MSN_SCAN_WINDOW[0]


###############################################################################################################
# Formula samplers
###############################################################################################################

class FormulaSampler(object):
    """
    Base class for formula sampler
    """

    def __init__(self, min_mz=MIN_MZ, max_mz=MAX_MZ):
        self.min_mz = min_mz
        self.max_mz = max_mz

    def sample(self, n_formulas):
        raise NotImplementedError


class DatabaseFormulaSampler(FormulaSampler):
    """
    A sampler to draw formula from a database
    """

    def __init__(self, database, min_mz=MIN_MZ, max_mz=MAX_MZ):
        """
        Initiliases database formula sampler
        :param database: a list of Formula objects containing chemical formulae from e.g. HMDB
        """
        self.database = database
        self.min_mz = min_mz
        self.max_mz = max_mz

    def sample(self, n_formulas):
        """
        Samples n_formulas from the specified database
        :param n_formulas: the number of formula to draw
        :param min_mz: minimum m/z of formula
        :param max_mz: maximum m/z of formula
        :return: a list of Formula objects
        """
        # filter database formulae to be within mz_range
        offset = 20  # to ensure that we have room for at least M+H
        formulas = list(set([(x.chemical_formula, x.name) for x in self.database]))
        sub_formulas = list(
            filter(lambda x: Formula(x[0]).mass >= self.min_mz and Formula(x[0]).mass <= self.max_mz - offset, formulas))
        logger.debug('{} unique formulas in filtered database'.format(len(sub_formulas)))
        chosen_formula_positions = np.random.choice(len(sub_formulas), size=n_formulas, replace=False)
        logger.debug('Sampled formulas')
        return [(Formula(sub_formulas[f][0]), sub_formulas[f][1]) for f in chosen_formula_positions]


class UniformMZFormulaSampler(FormulaSampler):
    """
    A sampler to generate formula uniformly between min_mz to max_mz, so just mz rather then formulas.
    Resulting in UnknownChemical objects instead of known_chemical ones.
    """

    def sample(self, n_formulas):
        """
        Samples n_formulas uniformly between min_mz and max_mz
        :param n_formulas: the number of formula to draw
        :param min_mz: minimum m/z of formula
        :param max_mz: maximum m/z of formula
        :return: a list of Formula objects
        """
        mz_list = np.random.rand(n_formulas) * (self.max_mz - self.min_mz) + self.min_mz
        return [(DummyFormula(m), None) for m in mz_list]


class PickEverythingFormulaSampler(DatabaseFormulaSampler):
    """
    A sampler that returns everything in the database
    """

    def __init__(self, database, min_mz=MIN_MZ, max_mz=MAX_MZ):
        """
        Initiliases database formula sampler
        :param database: a list of Formula objects containing chemical formulae from e.g. HMDB
        """
        self.database = database
        self.min_mz = min_mz
        self.max_mz = max_mz

    def sample(self, n_formulas):
        """
        Just return everything from the database
        :param n_formulas: ignored?
        :return: all formulae from the database
        """
        formula_list = [(Formula(x.chemical_formula), x.name) for x in self.database]
        return list(filter(lambda x: x[0].mass >= self.min_mz and x[0].mass <= self.max_mz, formula_list))


class EvenMZFormulaSampler(FormulaSampler):
    """
    A sampler that picks mz values evenly spaced, starting from where it left off
    Useful for test cases
    """

    def __init__(self):
        self.n_sampled = 0
        self.step = 100

    def sample(self, n_formulas):
        mz_list = []
        for i in range(n_formulas):
            new_mz = (self.n_sampled + 1) * self.step
            mz_list.append(new_mz)
            self.n_sampled += 1
        return [(DummyFormula(m), None) for m in mz_list]

class MZMLFormulaSampler(FormulaSampler):
    def __init__(self, mzml_file_name, source_polarity=POSITIVE):
        self.mzml_file_name = mzml_file_name
        self.source_polarity = source_polarity
        self._get_distributions()

    def _get_distributions(self):
        mzml_file_object = MZMLFile(str(self.mzml_file_name))
        mz_bins = {}
        for scan in mzml_file_object.scans:
            if not scan.ms_level == 1:
                continue
            for mz, intensity in scan.peaks:
                if self.source_polarity == POSITIVE:
                    mz -= PROTON_MASS
                elif self.source_polarity == NEGATIVE:
                    mz += PROTON_MASS
                else:
                    logger.warning("Unknown source polarity: {}".format(self.source_polarity))
                mz_bin = int(mz)
                if not mz_bin in mz_bins:
                    mz_bins[mz_bin] = intensity
                else:
                    mz_bins[mz_bin] += intensity
        total_intensity = sum(mz_bins.values())
        self.mz_bins = [(k, k+1) for k in mz_bins.keys()]
        self.mz_probs = [v/total_intensity for v in mz_bins.values()]

    def sample(self, n_formulas):
        mz_list = []
        for i in range(n_formulas):
            mz_bin_idx = np.random.choice(len(self.mz_bins), p=self.mz_probs)
            mz_bin = self.mz_bins[mz_bin_idx]
            mz = np.random.rand() * (mz_bin[1] - mz_bin[0]) + mz_bin[0]
            mz_list.append(mz)
        return [(DummyFormula(m), None) for m in mz_list]

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

class MZMLRTandIntensitySampler(RTAndIntensitySampler):
    def __init__(self, mzml_file_name, n_intensity_bins=10, min_rt=0, max_rt=1600, min_log_intensity=np.log(1e4), max_log_intensity=np.log(1e7), roi_params=None):
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity
        self.mzml_file_name = mzml_file_name
        self.roi_params = roi_params
        self.n_intensity_bins = n_intensity_bins
        if self.roi_params is None:
            self.roi_params = RoiParams()
        self._get_distributions()

    def _get_distributions(self):
        mzml_file_object = MZMLFile(str(self.mzml_file_name))
        rt_bins = {}
        mz_bins = {}
        for scan in mzml_file_object.scans:
            if not scan.ms_level == 1:
                continue
            mz, i = zip(*scan.peaks)
            total_intensity = sum(i)
            rt = scan.rt_in_seconds
            if rt < self.min_rt or rt > self.max_rt:
                continue
            rt_bin = int(rt)
            if not rt_bin in rt_bins:
                rt_bins[rt_bin] = total_intensity
            else:
                rt_bins[rt_bin] += total_intensity
        total_intensity = sum(rt_bins.values())
        self.rt_bins = [(k,k+1) for k in rt_bins.keys()]
        self.rt_probs = [v/total_intensity for v in rt_bins.values()]
        
        good, junk = make_roi(str(self.mzml_file_name), mz_tol=self.roi_params.mz_tol, mz_units=self.roi_params.mz_units, \
                              min_length=self.roi_params.min_length, min_intensity=self.roi_params.min_intensity, \
                              start_rt=self.roi_params.start_rt, stop_rt=self.roi_params.stop_rt, length_units=self.roi_params.length_units, \
                              ms_level=self.roi_params.ms_level, skip=self.roi_params.skip)
        log_roi_intensities = [np.log(max(r.intensity_list)) for r in good]
        log_roi_intensities = list(filter(lambda x: x >= self.min_log_intensity and x <= self.max_log_intensity, log_roi_intensities))
        hist, bin_edges = np.histogram(log_roi_intensities, bins=self.n_intensity_bins)
        total_i = hist.sum()
        hist = [h/total_i for h in hist]

        self.intensity_bins = [(b, bin_edges[i+1]) for i, b in enumerate(bin_edges[:-1])]
        self.intensity_probs = [h for h in hist]

    def sample(self, formula):
        rt_bin_idx = np.random.choice(len(self.rt_bins), p=self.rt_probs)
        rt_bin = self.rt_bins[rt_bin_idx]
        rt = np.random.rand() * (rt_bin[1] - rt_bin[0]) + rt_bin[0]

        intensity_bin_idx = np.random.choice(len(self.intensity_bins), p=self.intensity_probs)
        intensity_bin = self.intensity_bins[intensity_bin_idx]
        log_intensity = np.random.rand() * (intensity_bin[1] - intensity_bin[0]) + intensity_bin[0]
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
        assert sigma > 0
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


class ConstantChromatogramSampler(ChromatogramSampler):
    """
    A sampler to return constant chromatograms -- direct infusion
    """

    def sample(self, formula, rt, intensity):
        return ConstantChromatogram()

class MZMLChromatogramSampler(ChromatogramSampler):
    def __init__(self, mzml_file_name, roi_params = None):
        self.mzml_file_name = mzml_file_name
        self.roi_params = roi_params
        if self.roi_params is None:
            self.roi_params = RoiParams()
        
        self.good_rois = self._extract_rois()

    def _extract_rois(self):
        good, junk = make_roi(str(self.mzml_file_name), mz_tol=self.roi_params.mz_tol, mz_units=self.roi_params.mz_units, \
                                min_length=self.roi_params.min_length, min_intensity=self.roi_params.min_intensity, \
                                start_rt=self.roi_params.start_rt, stop_rt=self.roi_params.stop_rt, length_units=self.roi_params.length_units, \
                                ms_level=self.roi_params.ms_level, skip=self.roi_params.skip)
        logger.debug("Extracted {} good ROIs from {}".format(len(good), self.mzml_file_name))
        return good
    
    def sample(self, formula, rt, intensity):
        roi_idx = np.random.choice(len(self.good_rois))
        r = self.good_rois[roi_idx]
        chromatogram = EmpiricalChromatogram(np.array(r.rt_list), np.array(r.mz_list), \
                                            np.array(r.intensity_list), single_point_length=0.9)
        return chromatogram




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

    def __init__(self, poiss_peak_mean=10, min_mz=MIN_MZ_MS2, min_proportion=0.1, max_proportion=0.8):
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

    def sample(self, chemical):
        """
        Samples n_peaks of MS2 peaks uniformly between min_mz and the exact mass of the formula.
        The intensity is also randomly sampled between between min_proportion and max_proportion of the parent
        formula intensity
        :param formula: the parent formula
        :return: a tuple of (mz_list, intensity_list, parent_proportion)
        """
        n_peaks = np.random.poisson(self.poiss_peak_mean)
        max_mz = chemical.mass
        mz_list = uniform_list(n_peaks, self.min_mz, max_mz)
        intensity_list = uniform_list(n_peaks, 0, 1)

        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                            self.min_proportion

        return mz_list, intensity_list, parent_proportion

class FixedMS2Sampler(MS2Sampler):
    """
    Generates n_frags fragments, where each is chemical - i*10 mz
    """
    def __init__(self, n_frags=2):
        self.n_frags = n_frags

    def sample(self, chemical):
        initial_mz = chemical.mass
        mz_list = []
        intensity_list = []
        parent_proportion = 0.5
        for i in range(self.n_frags):
            mz_list.append(initial_mz - (i+1)*10)
            intensity_list.append(1)
        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        return mz_list, intensity_list, parent_proportion
        




class CRPMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 peaks following the CRP.
    """

    def __init__(self, n_draws=1000, min_mz=MIN_MZ_MS2, min_proportion=0.1, max_proportion=0.8, alpha=1,
                 base='uniform'):
        self.n_draws = n_draws
        self.min_mz = min_mz
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.alpha = alpha
        assert self.alpha > 0
        self.base = base
        assert self.base == 'uniform'

    def sample(self, chemical):
        max_mz = chemical.mass
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
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8, max_peaks=0, replace=False, id_field="SPECTRUMID"):
        self.mgf_file = mgf_file
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.replace = replace  # sample with replacement

        # load the mgf
        self.spectra_dict = load_mgf(self.mgf_file, id_field=id_field)

        # turn into a list where the last item is the number of times this one has been sampled
        self.spectra_list = [[s.precursor_mz, s, 0] for s in self.spectra_dict.values()]

        # filter to remove those with more than  max_peaks (if max_peaks > 0)
        if max_peaks > 0:
            self.spectra_list = list(filter(lambda x: len(x[1].peaks) <= max_peaks, self.spectra_list))

        # sort by precursor mz
        self.spectra_list.sort(key=lambda x: x[0])
        logger.debug("Loaded {} spectra from {}".format(len(self.spectra_list), self.mgf_file))

    def sample(self, chemical):
        formula_mz = chemical.mass
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


class ExactMatchMS2Sampler(MGFMS2Sampler):
    # to be completed. Where we have particular formulas and we
    # have a particular spectrum for each exact formula...
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8, id_field="SPECTRUMID"):
        super().__init__(mgf_file,min_proportion=min_proportion, max_proportion=max_proportion, id_field=id_field)
    def sample(self, chemical):
        spectrum = self.spectra_dict[chemical.database_accession]
        mz_list, intensity_list = zip(*spectrum.peaks)
        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                    self.min_proportion
        return mz_list, intensity_list, parent_proportion

class MZMLMS2Sampler(MS2Sampler):
    def __init__(self, mzml_file, min_n_peaks = 1, min_total_intensity=1e3, min_proportion=0.1, max_proportion=0.8):
        self.mzml_file_name = mzml_file
        self.mzml_object = MZMLFile(str(mzml_file))
        self.min_n_peaks = min_n_peaks
        self.min_total_intensity = min_total_intensity

        self.min_proportion = min_proportion
        self.max_proportion = max_proportion

        # only keep MS2 scans that have a least min_n_peaks and a total intesity of at least min_total_intesity
        self._filter_scans()

    def _filter_scans(self):
        ms2_scans = list(filter(lambda x: x.ms_level == 2 and
                                          len(x.peaks) >= self.min_n_peaks and 
                                          sum([i for mz,i in x.peaks]) >= self.min_total_intensity, self.mzml_object.scans))
        assert len(ms2_scans) > 0
        logger.debug("{} MS2 scansn remaining".format(len(ms2_scans)))
        self.ms2_scans = ms2_scans

    def sample(self,chemical):
        assert len(self.ms2_scans) > 0
        # pick a scan and removoe
        scan_idx = np.random.choice(len(self.ms2_scans),1)[0]
        scan = self.ms2_scans[scan_idx]
        del self.ms2_scans[scan_idx]

        parent_proportion = np.random.rand() * (self.max_proportion - self.min_proportion) + \
                            self.min_proportion
        
        mz_list, intensity_list = zip(*scan.peaks)

        return mz_list, intensity_list, parent_proportion
