"""
Sampling classes for ChemicalMixtureCreator
"""

from abc import ABC, abstractmethod

import numpy as np
from loguru import logger
from mass_spec_utils.data_import.mzml import MZMLFile
from mass_spec_utils.library_matching.gnps import load_mgf

from vimms.Chromatograms import FunctionalChromatogram, ConstantChromatogram, \
    EmpiricalChromatogram
from vimms.Common import Formula, DummyFormula, uniform_list, \
    DEFAULT_MS1_SCAN_WINDOW, DEFAULT_MSN_SCAN_WINDOW, \
    POSITIVE, NEGATIVE, PROTON_MASS, DEFAULT_SCAN_TIME_DICT
from vimms.Roi import make_roi, RoiBuilderParams

MIN_MZ = DEFAULT_MS1_SCAN_WINDOW[0]
MAX_MZ = DEFAULT_MS1_SCAN_WINDOW[1]
MIN_MZ_MS2 = DEFAULT_MSN_SCAN_WINDOW[0]


###############################################################################
# Formula samplers
###############################################################################

class FormulaSampler(ABC):
    """
    Base class for formula sampler
    """

    def __init__(self, min_mz=MIN_MZ, max_mz=MAX_MZ):
        """
        Create a Formula sampler
        Args:
            min_mz: the minimum m/z value of formulae to sample from
            max_mz: the maximum m/z value of formulae to sample from
        """
        self.min_mz = min_mz
        self.max_mz = max_mz

    @abstractmethod
    def sample(self, n_formulas):
        pass


class DatabaseFormulaSampler(FormulaSampler):
    """
    A sampler to draw formula from a database
    """

    def __init__(self, database, min_mz=MIN_MZ, max_mz=MAX_MZ):
        """
        Initiliases database formula sampler

        Args:
            database: a list of Formula objects containing chemical
                      formulae from e.g. HMDB
            min_mz: the minimum m/z value of formulae to sample from
            max_mz: the maximum m/z value of formulae to sample from
        """
        super().__init__(min_mz=min_mz, max_mz=max_mz)
        self.database = database

    def sample(self, n_formulas):
        """
        Samples n_formulas from the specified database

        Args:
            n_formulas: the number of formula to draw

        Returns: a list of Formula objects

        """
        # filter database formulae to be within mz_range
        offset = 20  # to ensure that we have room for at least M+H
        formulas = list(
            set([(x.chemical_formula, x.name) for x in self.database]))
        sub_formulas = list(
            filter(lambda x: Formula(x[0]).mass >= self.min_mz and Formula(
                x[0]).mass <= self.max_mz - offset,
                   formulas))
        logger.debug(
            '{} unique formulas in filtered database'.format(
                len(sub_formulas)))
        chosen_formula_positions = np.random.choice(len(sub_formulas),
                                                    size=n_formulas,
                                                    replace=False)
        logger.debug('Sampled formulas')
        return [(Formula(sub_formulas[f][0]), sub_formulas[f][1]) for f in
                chosen_formula_positions]


class UniformMZFormulaSampler(FormulaSampler):
    """
    A sampler to generate formula uniformly between min_mz to max_mz, so
    just mz rather then formulas. Resulting in UnknownChemical objects
    instead of known_chemical ones.
    """

    def sample(self, n_formulas):
        """
        Samples n_formulas uniformly between min_mz and max_mz

        Args:
            n_formulas: the number of formula to draw

        Returns: a list of Formula objects

        """
        mz_list = np.random.rand(n_formulas) * (
                self.max_mz - self.min_mz) + self.min_mz
        return [(DummyFormula(m), None) for m in mz_list]


class PickEverythingFormulaSampler(DatabaseFormulaSampler):
    """
    A sampler that returns everything in the database
    """

    def __init__(self, database, min_mz=MIN_MZ, max_mz=MAX_MZ):
        """
        Initiliases a Pick-Everything formula sampler

        Args:
            database: a list of Formula objects containing chemical
                      formulae from e.g. HMDB
            min_mz: the minimum m/z value of formulae to sample from
            max_mz: the maximum m/z value of formulae to sample from
        """
        super().__init__(min_mz=min_mz, max_mz=max_mz)
        self.database = database

    def sample(self, n_formulas):
        """
        Just return everything from the database

        Args:
            n_formulas: ignored?

        Returns: all formulae from the database

        """
        formula_list = [(Formula(x.chemical_formula), x.name) for x in
                        self.database]
        return list(filter(
            lambda x: x[0].mass >= self.min_mz and x[0].mass <= self.max_mz,
            formula_list))


class EvenMZFormulaSampler(FormulaSampler):
    """
    A sampler that picks mz values evenly spaced, starting from where
    it left off. Useful for test cases
    """

    def __init__(self):
        """
        Create an even m/z formula sampler
        """
        self.n_sampled = 0
        self.step = 100

    def sample(self, n_formulas):
        """
        Sample up to n_formulas from this sampler
        Args:
            n_formulas: the number of formula to return

        Returns: the list of formulae having evenly spaced m/z values

        """
        mz_list = []
        for i in range(n_formulas):
            new_mz = (self.n_sampled + 1) * self.step
            mz_list.append(new_mz)
            self.n_sampled += 1
        return [(DummyFormula(m), None) for m in mz_list]


class MZMLFormulaSampler(FormulaSampler):
    """
    A sampler to generate m/z values from a histogram of m/z taken from
    a user supplied mzML file
    """

    def __init__(self, mzml_file_name, min_mz=MIN_MZ, max_mz=MAX_MZ,
                 source_polarity=POSITIVE):
        """
        Create an mzML formula sampler
        Args:
            mzml_file_name: the source mzML file
            min_mz: the minimum m/z to consider
            max_mz: the maximum m/z to consider
            source_polarity: either POSITIVE or NEGATIVE
        """
        super().__init__(min_mz=min_mz, max_mz=max_mz)
        self.mzml_file_name = mzml_file_name
        self.source_polarity = source_polarity
        self._get_distributions()

    def _get_distributions(self):
        """
        Compute the distribution of m/z values by placing them into bins
        Returns: None

        """
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
                    logger.warning("Unknown source polarity: {}".format(
                        self.source_polarity))
                if mz < self.min_mz or mz > self.max_mz:
                    continue
                mz_bin = int(mz)
                if mz_bin not in mz_bins:
                    mz_bins[mz_bin] = intensity
                else:
                    mz_bins[mz_bin] += intensity
        total_intensity = sum(mz_bins.values())
        self.mz_bins = [(k, k + 1) for k in mz_bins.keys()]
        self.mz_probs = [v / total_intensity for v in mz_bins.values()]

    def sample(self, n_formulas):
        """
        Sample up to n_formulas from the m/z values in the mzML file
        Args:
            n_formulas: the number of formula to sample

        Returns: a list of Formula objects

        """
        mz_list = []
        for i in range(n_formulas):
            mz_bin_idx = np.random.choice(len(self.mz_bins), p=self.mz_probs)
            mz_bin = self.mz_bins[mz_bin_idx]
            mz = np.random.rand() * (mz_bin[1] - mz_bin[0]) + mz_bin[0]
            mz_list.append(mz)
        return [(DummyFormula(m), None) for m in mz_list]


###############################################################################
# Samplers for RT and intensity when initialising a Formula
###############################################################################


class RTAndIntensitySampler(ABC):
    """
    Base class for RT and intensity sampler. Usually used when initialising
    a formula object.
    """

    @abstractmethod
    def sample(self, formula):
        pass


class UniformRTAndIntensitySampler(RTAndIntensitySampler):
    """
    A sampler to sample RT and log intensity uniformly.
    See class def for min and max log intensity.
    Returns actual intensity, but samples in log space.
    """

    def __init__(self, min_rt=0, max_rt=1600, min_log_intensity=np.log(1e4),
                 max_log_intensity=np.log(1e7)):
        """
        Initialises uniform RT and intensity sampler

        Args:
            min_rt: minimum RT
            max_rt: maximum RT
            min_log_intensity: minimum log intensity
            max_log_intensity: maximum log intensity
        """
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity

    def sample(self, formula):
        """
        Samples RT and log intensity uniformly between (min_rt, max_rt) and
        (min_log_intensity, max_log_intensity)

        Args:
            formula: the formula to condition on (can be ignored)

        Returns: a tuple of (RT, intensity)

        """
        rt = np.random.rand() * (self.max_rt - self.min_rt) + self.min_rt
        diff = self.max_log_intensity - self.min_log_intensity
        log_intensity = np.random.rand() * (diff) + self.min_log_intensity
        return rt, np.exp(log_intensity)


class MZMLRTandIntensitySampler(RTAndIntensitySampler):
    """
    A sampler to sample RT and intensity values from an existing mzML file.
    Useful to mimic the characteristics of actual experimental data.
    """
    def __init__(self, mzml_file_name, n_intensity_bins=10, min_rt=0,
                 max_rt=1600, min_log_intensity=np.log(1e4),
                 max_log_intensity=np.log(1e7), roi_params=None):
        """
        Create an instance of MZMLRTandIntensitySampler.
        Args:
            mzml_file_name: the source mzML filename
            n_intensity_bins: number of bins for intensities
            min_rt: the minimum RT to consider
            max_rt: the maximum RT to consider
            min_log_intensity: the minimum intensity (in log) to consider
            max_log_intensity: the maximum intensity (in log) to consider
            roi_params: parameters for ROI building, as defined in [vimms.Roi.RoiBuilderParams][].
        """
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.min_log_intensity = min_log_intensity
        self.max_log_intensity = max_log_intensity
        self.mzml_file_name = mzml_file_name
        self.roi_params = roi_params
        self.n_intensity_bins = n_intensity_bins
        if self.roi_params is None:
            self.roi_params = RoiBuilderParams()
        self._get_distributions()

    def _get_distributions(self):
        """
        Compute distributions of RT and intensity values from the mzML file.

        Returns:None

        """
        mzml_file_object = MZMLFile(str(self.mzml_file_name))
        rt_bins = {}
        # mz_bins = {}
        for scan in mzml_file_object.scans:
            if not scan.ms_level == 1:
                continue
            mz, i = zip(*scan.peaks)
            total_intensity = sum(i)
            rt = scan.rt_in_seconds
            if rt < self.min_rt or rt > self.max_rt:
                continue
            rt_bin = int(rt)
            if rt_bin not in rt_bins:
                rt_bins[rt_bin] = total_intensity
            else:
                rt_bins[rt_bin] += total_intensity
        total_intensity = sum(rt_bins.values())
        self.rt_bins = [(k, k + 1) for k in rt_bins.keys()]
        self.rt_probs = [v / total_intensity for v in rt_bins.values()]

        good = make_roi(str(self.mzml_file_name), self.roi_params)
        log_roi_intensities = [np.log(max(r.intensity_list)) for r in good]
        log_roi_intensities = filter(
            lambda x: self.min_log_intensity <= x <= self.max_log_intensity,
            log_roi_intensities
        )
        log_roi_intensities = list(log_roi_intensities)
        hist, bin_edges = np.histogram(log_roi_intensities,
                                       bins=self.n_intensity_bins)
        total_i = hist.sum()
        hist = [h / total_i for h in hist]

        self.intensity_bins = [(b, bin_edges[i + 1]) for i, b in
                               enumerate(bin_edges[:-1])]
        self.intensity_probs = [h for h in hist]

    def sample(self, formula):
        """
        Sample RT and intensity value from this sampler
        Args:
            formula: the chemical formula, unused for now.

        Returns: a tuple of (RT, intensity) values.

        """
        rt_bin_idx = np.random.choice(len(self.rt_bins), p=self.rt_probs)
        rt_bin = self.rt_bins[rt_bin_idx]
        rt = np.random.rand() * (rt_bin[1] - rt_bin[0]) + rt_bin[0]

        intensity_bin_idx = np.random.choice(len(self.intensity_bins),
                                             p=self.intensity_probs)
        intensity_bin = self.intensity_bins[intensity_bin_idx]
        log_intensity = np.random.rand() * (
                intensity_bin[1] - intensity_bin[0]) + intensity_bin[0]
        return rt, np.exp(log_intensity)


###############################################################################
# Chromatogram samplers
###############################################################################


class ChromatogramSampler(ABC):
    """
    Base class for chromatogram sampler.
    """

    @abstractmethod
    def sample(self, formula, rt, intensity):
        pass


class GaussianChromatogramSampler(ChromatogramSampler):
    """
    A sampler to return Gaussian-shaped chromatogram
    """

    def __init__(self, sigma=10):
        """
        Create a Gaussian-shaped chromatogram sampler
        Args:
            sigma: parameter for the Gaussian distribution to sample from
        """
        assert sigma > 0
        self.sigma = sigma

    def sample(self, formula, rt, intensity):
        """
        Sample a Gaussian-shaped chromatogram

        Args:
            formula: the formula to condition on (can be ignored)
            rt: RT to condition on (can be ignored)
            intensity: intensity to condition on (can be ignored)

        Returns: a [vimms.Chromatograms.FunctionalChromatogram] object.

        """
        return FunctionalChromatogram('normal', [0, self.sigma])


class ConstantChromatogramSampler(ChromatogramSampler):
    """
    A sampler to return constant chromatograms -- direct infusion
    """

    def sample(self, formula, rt, intensity):
        """
        Sample a constant chromatogram (present everywhere)
        Args:
            formula: formula, unused
            rt: RT, unused
            intensity: intensity, unused

        Returns: a [vimms.Chromatograms.ConstantChromatogram] object.

        """
        return ConstantChromatogram()


class MZMLChromatogramSampler(ChromatogramSampler):
    """
    A sampler to return chromatograms extracted from an existing mzML file.
    Useful to mimic the characteristics of actual experimental data.
    """
    def __init__(self, mzml_file_name, roi_params=None):
        """
        Create an MZMLChromatogramSampler object.
        Args:
            mzml_file_name: the input mzML file.
            roi_params: parameters for ROI building, as defined in [vimms.Roi.RoiBuilderParams][].
        """
        self.mzml_file_name = mzml_file_name
        self.roi_params = roi_params
        if self.roi_params is None:
            self.roi_params = RoiBuilderParams()

        self.good_rois = self._extract_rois()

    def _extract_rois(self):
        """
        Extract regions-of-interests from the mzML file

        Returns: the list of good ROIs that have been filtered according to certain criteria.

        """
        good = make_roi(str(self.mzml_file_name), self.roi_params)
        logger.debug("Extracted {} good ROIs from {}".format(
            len(good), self.mzml_file_name))
        return good

    def sample(self, formula, rt, intensity):
        """
        Sample an empirical chromatogram extracted from the mzML file
        Args:
            formula: formula, unused
            rt: RT, unused
            intensity: intensity, unused

        Returns: a [vimms.Chromatograms.EmpiricalChromatogram] object.

        """
        roi_idx = np.random.choice(len(self.good_rois))
        r = self.good_rois[roi_idx]
        chromatogram = EmpiricalChromatogram(np.array(r.rt_list),
                                             np.array(r.mz_list),
                                             np.array(r.intensity_list),
                                             single_point_length=0.9)
        return chromatogram


###############################################################################
# MS2 samplers
###############################################################################


class MS2Sampler(ABC):
    """
    Base class for MS2 sampler
    """

    @abstractmethod
    def sample(self, formula):
        pass


class UniformMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 peaks uniformly between min_mz and
    the mass of the formula.
    """

    def __init__(self, poiss_peak_mean=10, min_mz=MIN_MZ_MS2,
                 min_proportion=0.1, max_proportion=0.8):
        """
        Initialises uniform MS2 sampler

        Args:
            poiss_peak_mean: the mean of the Poisson distribution used
                             to draw the number of peaks
            min_mz: minimum m/z value
            min_proportion: minimum proportion from the parent MS1
                            peak intensities
            max_proportion: maximum proportion from the parent MS1
                            peak intensities
        """
        self.poiss_peak_mean = poiss_peak_mean
        self.min_mz = min_mz
        # proportion of parent intensity shared by MS2
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion

    def sample(self, chemical):
        """
        Samples n_peaks of MS2 peaks uniformly between min_mz and
        the exact mass of the formula. The intensity is also randomly sampled
        between between min_proportion and max_proportion of the parent
        formula intensity

        Args:
            chemical: the chemical to compute max m/z value from

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """
        n_peaks = np.random.poisson(self.poiss_peak_mean)
        max_mz = chemical.mass
        mz_list = uniform_list(n_peaks, self.min_mz, max_mz)
        intensity_list = uniform_list(n_peaks, 0, 1)

        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        parent_proportion = np.random.rand() * (
                    self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion


class FixedMS2Sampler(MS2Sampler):
    """
    Generates n_frags fragments, where each is chemical - i*10 mz
    """

    def __init__(self, n_frags=2):
        """
        Create a fixed MS2 sampler
        Args:
            n_frags: the number of fragment peaks to generate
        """
        self.n_frags = n_frags

    def sample(self, chemical):
        """
        Sample MS2 spectra using chemical as the parent
        Args:
            chemical: the parent chemical

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """
        initial_mz = chemical.mass
        mz_list = []
        intensity_list = []
        parent_proportion = 0.5
        for i in range(self.n_frags):
            mz_list.append(initial_mz - (i + 1) * 10)
            intensity_list.append(1)
        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        return mz_list, intensity_list, parent_proportion


class CRPMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 peaks following the Chinese Restaurant Process (CRP),
    i.e. an MS2 peak that has been selected in one spectra has a higher likelihood to appear
    again elsewhere.
    """

    def __init__(self, n_draws=1000, min_mz=MIN_MZ_MS2, min_proportion=0.1,
                 max_proportion=0.8, alpha=1,
                 base='uniform'):
        """
        Create a CRP-based MS2 sampler.
        Args:
            n_draws: the number of draws from the CRP process
            min_mz: the minimum m/z value to consider
            min_proportion: the minimum proportion to consider
            max_proportion: the maximum proportion to consider
            alpha: CRP parameter
            base: base distribution for the CRP process
        """
        self.n_draws = n_draws
        self.min_mz = min_mz
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.alpha = alpha
        assert self.alpha > 0
        self.base = base
        assert self.base == 'uniform'

    def sample(self, chemical):
        """
        Sample MS2 spectra using chemical as the parent
        Args:
            chemical: the parent chemical

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """

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
        parent_proportion = \
            np.random.rand() * (self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion

    def _base_sample(self, max_mz):
        return np.random.rand() * (max_mz - self.min_mz) + self.min_mz


class MGFMS2Sampler(MS2Sampler):
    """
    A sampler that generates MS2 spectra from real ones defined in some MGF file.
    """
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8,
                 max_peaks=0, replace=False,
                 id_field="SPECTRUMID"):
        """
        Create an MGFMS2Sampler object.
        Args:
            mgf_file: input MGF file.
            min_proportion: the minimum proportion to consider
            max_proportion: the maximum proportion to consider
            max_peaks: the maximum number of peaks
            replace: whether to sample with replacement or not
            id_field: the ID field in the MGF file
        """
        self.mgf_file = mgf_file
        self.min_proportion = min_proportion
        self.max_proportion = max_proportion
        self.replace = replace  # sample with replacement

        # load the mgf
        self.spectra_dict = load_mgf(self.mgf_file, id_field=id_field)

        # turn into a list where the last item is the number of times
        # this one has been sampled
        self.spectra_list = [[s.precursor_mz, s, 0] for s in
                             self.spectra_dict.values()]

        # filter to remove those with more than  max_peaks (if max_peaks > 0)
        if max_peaks > 0:
            self.spectra_list = list(
                filter(lambda x: len(x[1].peaks) <= max_peaks,
                       self.spectra_list))

        # sort by precursor mz
        self.spectra_list.sort(key=lambda x: x[0])
        logger.debug("Loaded {} spectra from {}".format(len(self.spectra_list),
                                                        self.mgf_file))

    def sample(self, chemical):
        """
        Sample MS2 spectra using chemical as the parent
        Args:
            chemical: the parent chemical

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """

        formula_mz = chemical.mass
        sub_spec = list(filter(lambda x: x[0] < formula_mz, self.spectra_list))
        if len(sub_spec) == 0:
            # if there aren't any smaller than the mz, we just take any one
            sub_spec = self.spectra_list

        # sample one. If replace == True we take any, if not we only
        # take those that have not been sampled before
        found_permissable = False
        n_attempts = 0
        while not found_permissable:
            n_attempts += 1
            spec = np.random.choice(len(sub_spec))
            if self.replace is True or sub_spec[spec][2] == 0 or n_attempts > 100:
                found_permissable = True

        sub_spec[spec][2] += 1  # add one to the count
        spectrum = sub_spec[spec][1]
        mz_list, intensity_list = zip(*spectrum.peaks)
        s = sum(intensity_list)
        intensity_list = [i / s for i in intensity_list]
        parent_proportion = np.random.rand() * (
                    self.max_proportion - self.min_proportion) + self.min_proportion

        return mz_list, intensity_list, parent_proportion


class ExactMatchMS2Sampler(MGFMS2Sampler):
    """
    Exact match MS2 sampler allows us to have particular formulas and we
    have a particular spectrum for each exact formula...

    TODO: not sure if this class is actually completed and fully tested.
    """
    def __init__(self, mgf_file, min_proportion=0.1, max_proportion=0.8,
                 id_field="SPECTRUMID"):
        super().__init__(mgf_file, min_proportion=min_proportion,
                         max_proportion=max_proportion, id_field=id_field)

    def sample(self, chemical):
        """
        Sample MS2 spectra using chemical as the parent
        Args:
            chemical: the parent chemical

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """

        spectrum = self.spectra_dict[chemical.database_accession]
        mz_list, intensity_list = zip(*spectrum.peaks)
        parent_proportion = np.random.rand() * (
                    self.max_proportion - self.min_proportion) + self.min_proportion
        return mz_list, intensity_list, parent_proportion


class MZMLMS2Sampler(MS2Sampler):
    """
    A sampler that sample MS2 spectra from an actual mzML file.
    """
    def __init__(self, mzml_file, min_n_peaks=1, min_total_intensity=1e3,
                 min_proportion=0.1, max_proportion=0.8,
                 with_replacement=False):
        """
        Create an MZMLMS2Sampler object
        Args:
            mzml_file: the source mzML file
            min_n_peaks: the minimum number of peaks to consider for each frag. spectra
            min_total_intensity: the minimum total intensity
            min_proportion: the minimum proportion to consider
            max_proportion: the maximum proportion to consider
            with_replacement: whether to sample with replacement or not
        """
        self.mzml_file_name = mzml_file
        self.mzml_object = MZMLFile(str(mzml_file))
        self.min_n_peaks = min_n_peaks
        self.min_total_intensity = min_total_intensity
        self.with_replacement = with_replacement

        self.min_proportion = min_proportion
        self.max_proportion = max_proportion

        # only keep MS2 scans that have a least min_n_peaks and
        # a total intesity of at least min_total_intesity
        self._filter_scans()

    def _filter_scans(self):
        """
        Filters MS2 scans according to certain criteria

        Returns: None

        """
        ms2_scans = list(filter(
            lambda x: x.ms_level == 2 and len(x.peaks) >= self.min_n_peaks and sum(
                [i for mz, i in x.peaks]) >= self.min_total_intensity, self.mzml_object.scans))
        assert len(
            ms2_scans) > 0, "After filtering no ms2 scans remain - " \
                            "consider loosening filter parameters"
        logger.debug("{} MS2 scansn remaining".format(len(ms2_scans)))
        self.ms2_scans = ms2_scans

    def sample(self, chemical):
        """
        Sample MS2 spectra using chemical as the parent
        Args:
            chemical: the parent chemical

        Returns: a tuple of (mz_list, intensity_list, parent_proportion)

        """

        assert len(
            self.ms2_scans) > 0, "MS2 sampler ran out of scans. " \
                                 "Consider an alternative, or " \
                                 "setting with_replacement to True"
        # pick a scan and removoe
        scan_idx = np.random.choice(len(self.ms2_scans), 1)[0]
        scan = self.ms2_scans[scan_idx]
        if not self.with_replacement:
            del self.ms2_scans[scan_idx]

        parent_proportion = np.random.rand() * (
                self.max_proportion - self.min_proportion) + self.min_proportion

        mz_list, intensity_list = zip(*scan.peaks)

        return mz_list, intensity_list, parent_proportion


###############################################################################
# Scan time samplers
###############################################################################

class ScanTimeSampler(ABC):
    """
    Base class for scan time sampler
    """

    @abstractmethod
    def sample(self, current_level, next_level):
        pass


class DefaultScanTimeSampler(ScanTimeSampler):
    """
    A scan time sampler that returns some fixed values that represent the average scan times for
    MS1 and MS2 scans.
    """

    def __init__(self, scan_time_dict=None):
        """
        Initialises a default scan time sampler object.

        Args:
            scan_time_dict: A dictionary of scan times for each MS-level.
                            It should look like this: {1: 0.4, 2: 0.2}.
                            If not specified, then the default value is used.
                            Note that this default is obtained from our Orbitrap instrument and
                            would certainly differ from yours!
        """

        self.scan_time_dict = scan_time_dict if scan_time_dict is not None \
            else DEFAULT_SCAN_TIME_DICT

    def sample(self, current_level, next_level):
        """
        Sample a scan duration given the MS levels of current and next scans.
        Args:
            current_level: the MS level of the current scan
            next_level: the MS level of the next scan

        Returns: a sampled scan duration value

        """
        return self.scan_time_dict[current_level]


class MzMLScanTimeSampler(ScanTimeSampler):
    """
    A scan time sampler that obtains its values from an existing MZML file.
    """

    def __init__(self, mzml_file, use_mean=True):
        """
        Initialises a MZML scan time sampler object.

        Args:
            mzml_file: the source MZML file
            use_mean: whether to store the scan times as distributions of values to sample
                      from, or as a single mean value
        """

        self.mzml_file = str(mzml_file)
        self.use_mean = use_mean

        self.time_dict = self._extract_timing(self.mzml_file)
        self.is_frag_file = self._is_frag_file(self.time_dict)
        self.mean_time_dict = self._extract_mean_time(self.time_dict,
                                                      self.is_frag_file)
        if self.is_frag_file and len(self.time_dict[(1, 1)]) == 0:
            # this could sometimes happen if there's not enough MS1 scan
            # followed by another MS1 scan
            default = DEFAULT_SCAN_TIME_DICT[1]
            logger.warning(
                'Not enough MS1 scans to compute (1, 1) scan duration. '
                'The default of %f will be used' % default)
            self.time_dict[(1, 1)] = [default]

    def sample(self, current_level, next_level):
        """
        Sample a scan duration given the MS levels of current and next scans.
        Args:
            current_level: the MS level of the current scan
            next_level: the MS level of the next scan

        Returns: a sampled scan duration value

        """

        if self.use_mean:
            # return only the average time for current_level
            return self.mean_time_dict[current_level]
        else:
            # sample a scan duration value extracted from the mzML based
            # on the current and next level
            values = self.time_dict[(current_level, next_level)]
            sampled = np.random.choice(values, replace=False, size=1)
            return sampled[0]

    def _extract_timing(self, seed_file):
        """
        Extracts timing information from a seed file

        Args:
            seed_file: The seed file in mzML format.
                       If it's a DDA file (containing MS1 and MS2 scans) then both MS1 and
                       MS2 timing will be extracted.
                       If it's only a fullscan file (containing MS1 scans) then only MS1
                       timing will be extracted.

        Returns: a dictionary of time information. Key should be the ms-level,
                 1 or 2, and value is the average time of scans at that level

        """
        logger.debug('Extracting timing dictionary from seed file')
        seed_mzml = MZMLFile(seed_file)

        time_dict = {(1, 1): [], (1, 2): [], (2, 1): [], (2, 2): []}
        for i, s in enumerate(seed_mzml.scans[:-1]):
            current = s.ms_level
            next_ = seed_mzml.scans[i + 1].ms_level
            tup = (current, next_)
            time_dict[tup].append(60 * seed_mzml.scans[
                i + 1].rt_in_minutes - 60 * s.rt_in_minutes)
        return time_dict

    def _is_frag_file(self, time_dict):
        """
        Checks that the time dictionary comes from a fragmentation file or not.
        Args:
            time_dict: a time dictionary

        Returns: True if it comes from a fragmentation file, False otherwise.

        """
        is_frag_file = False
        if (1, 2) in time_dict and len(time_dict[(1, 2)]) > 0 and \
                (2, 2) in time_dict and len(time_dict[(2, 2)]) > 0:
            # seed_file must contain timing on (1,2) and (2,2)
            # i.e. it must be a DDA file with MS1 and MS2 scans
            is_frag_file = True
        return is_frag_file

    def _extract_mean_time(self, time_dict, is_frag_file):
        """
        Construct mean timing dict in the right format for later use

        Args:
            time_dict: a timing dictionary
            is_frag_file: whether it's a fragmentation file or not

        Returns: the mean time dictionary

        """
        mean_time_dict = {}
        if is_frag_file:
            # extract ms1 and ms2 timing from fragmentation mzML
            for k, v in time_dict.items():
                if k == (1, 2):
                    key = 1
                elif k == (2, 2):
                    key = 2
                else:
                    continue

                mean = sum(v) / len(v)
                mean_time_dict[key] = mean
                logger.debug('%d: %f' % (key, mean))
            assert 1 in mean_time_dict and 2 in mean_time_dict
        else:
            # extract ms1 timing only from fullscan mzML
            key = 1
            v = time_dict[(1, 1)]
            mean = sum(v) / len(v)
            mean_time_dict[key] = mean
            logger.debug('%d: %f' % (key, mean))

        return mean_time_dict
