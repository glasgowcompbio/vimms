import copy
import math
from pathlib import Path

import numpy as np
import scipy
import scipy.stats
from loguru import logger

from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, GaussianChromatogramSampler, UniformMS2Sampler
from vimms.ChineseRestaurantProcess import Restricted_Crp
from vimms.Common import CHEM_DATA, POS_TRANSFORMATIONS, GET_MS2_BY_PEAKS, GET_MS2_BY_SPECTRA, load_obj, save_obj, \
    Formula, DummyFormula
from vimms.Noise import GaussianPeakNoise


class DatabaseCompound(object):
    def __init__(self, name, chemical_formula, monisotopic_molecular_weight, smiles, inchi, inchikey):
        self.name = name
        self.chemical_formula = chemical_formula
        self.monisotopic_molecular_weight = monisotopic_molecular_weight
        self.smiles = smiles
        self.inchi = inchi
        self.inchikey = inchikey


class Isotopes(object):
    def __init__(self, formula):
        self.formula = formula
        self.C12_proportion = 0.989
        self.mz_diff = 1.0033548378
        # TODO: Add functionality for elements other than Carbon

    def get_isotopes(self, total_proportion):
        peaks = [() for i in range(len(self._get_isotope_proportions(total_proportion)))]
        for i in range(len(peaks)):
            peaks[i] += (self._get_isotope_mz(self._get_isotope_names(i)),)
            peaks[i] += (self._get_isotope_proportions(total_proportion)[i],)
            peaks[i] += (self._get_isotope_names(i),)
        return peaks

    # outputs [(mz_1, intensity_proportion_1, isotope_name_1),...,(mz_n, intensity_proportion_n, isotope_name_n)]

    def _get_isotope_proportions(self, total_proportion):
        proportions = []
        while sum(proportions) < total_proportion:
            proportions.extend(
                [scipy.stats.binom.pmf(len(proportions), self.formula._get_n_element("C"), 1 - self.C12_proportion)])
        normalised_proportions = [proportions[i] / sum(proportions) for i in range(len(proportions))]
        return normalised_proportions

    def _get_isotope_names(self, isotope_number):
        if isotope_number == 0:
            return "Mono"
        else:
            return str(isotope_number) + "C13"

    def _get_isotope_mz(self, isotope):
        if isotope == "Mono":
            return self.formula._get_mz()
        elif isotope[-3:] == "C13":
            return self.formula._get_mz() + float(isotope.split("C13")[0]) * self.mz_diff
        else:
            return None


class Adducts(object):
    def __init__(self, formula, adduct_proportion_cutoff=0.05, adduct_prior_dict=None):
        if adduct_prior_dict is None:
            self.adduct_names = list(POS_TRANSFORMATIONS.keys())
            self.adduct_prior = np.ones(len(self.adduct_names)) * 0.1
            self.adduct_prior[0] = 1.0  # give more weight to the first one, i.e. M+H
        else:
            self.adduct_names = list(adduct_prior_dict.keys())
            self.adduct_prior = np.array(list(adduct_prior_dict.values()))
        self.formula = formula
        self.adduct_proportion_cutoff = adduct_proportion_cutoff

    def get_adducts(self):
        adducts = []
        proportions = self._get_adduct_proportions()
        for j in range(len(self.adduct_names)):
            if proportions[j] != 0:
                adducts.extend([(self._get_adduct_names()[j], proportions[j])])
        return adducts

    def _get_adduct_proportions(self):
        # TODO: replace this with something proper
        proportions = np.random.dirichlet(self.adduct_prior)
        while max(proportions) < 0.2:
            proportions = np.random.dirichlet(self.adduct_prior)
        proportions[np.where(proportions < self.adduct_proportion_cutoff)] = 0
        proportions = proportions / max(proportions)
        proportions.tolist()
        return proportions

    def _get_adduct_names(self):
        return self.adduct_names


class Chemical(object):

    def __repr__(self):
        raise NotImplementedError()


class UnknownChemical(Chemical):
    """
    Chemical from an unknown chemical formula
    """

    def __init__(self, mz, rt, max_intensity, chromatogram, children=None):
        self.max_intensity = max_intensity
        self.isotopes = [(mz, 1, "Mono")]  # [(mz, intensity_proportion, isotope,name)]
        self.adducts = [("M+H", 1)]
        self.rt = rt
        self.chromatogram = chromatogram
        self.children = children
        self.ms_level = 1
        self.mz_diff = 0
        self.mass = mz

    def __repr__(self):
        return 'UnknownChemical mz=%.4f rt=%.2f max_intensity=%.2f' % (
            self.isotopes[0][0], self.rt, self.max_intensity)

    def get_key(self):
        """
        Turns a chemical object into (mz, rt, intensity) tuples for equal comparison
        :return: a tuple of the three values
        """
        return (tuple(self.isotopes), self.rt, self.max_intensity)

    def __eq__(self, other):
        if not isinstance(other, UnknownChemical):
            return False
        return self.get_key() == other.get_key()

    def __hash__(self):
        return hash(self.get_key())


class KnownChemical(Chemical):
    """
    Chemical from a known chemical formula
    """

    def __init__(self, formula, isotopes, adducts, rt, max_intensity, chromatogram, children=None,
                 include_adducts_isotopes=True, total_proportion=0.99):
        self.formula = formula
        self.mz_diff = isotopes.mz_diff
        if include_adducts_isotopes == True:
            self.isotopes = isotopes.get_isotopes(total_proportion)
            self.adducts = adducts.get_adducts()
        else:
            mz = isotopes.get_isotopes(total_proportion)[0][0]
            self.isotopes = [(mz, 1, "Mono")]
            self.adducts = [("M+H", 1)]
        self.rt = rt
        self.max_intensity = max_intensity
        self.chromatogram = chromatogram
        self.children = children
        self.ms_level = 1
        self.mass = self.formula.mass

    def __repr__(self):
        return 'KnownChemical - %r rt=%.2f max_intensity=%.2f' % (
            self.formula.formula_string, self.rt, self.max_intensity)

    def __eq__(self, other):
        if not isinstance(other, KnownChemical):
            return False
        return self.formula.formula_string == other.formula.formula_string

    def __hash__(self):
        return hash(self.formula.formula_string)


class MSN(Chemical):
    """
    ms2+ fragments
    """

    def __init__(self, mz, ms_level, prop_ms2_mass, parent_mass_prop, children=None, parent=None):
        self.isotopes = [(mz, None, "MSN")]
        self.ms_level = ms_level
        self.prop_ms2_mass = prop_ms2_mass
        self.parent_mass_prop = parent_mass_prop
        self.children = children
        self.parent = parent

    def __repr__(self):
        return 'MSN Fragment mz=%.4f ms_level=%d' % (self.isotopes[0][0], self.ms_level)


class ChemicalCreator(object):
    def __init__(self, peak_sampler, ROI_sources=None, database=None):
        self.peak_sampler = peak_sampler
        self.ROI_sources = ROI_sources
        self.database = database

        # sort database compounds by their mass
        if self.database is not None:
            logger.debug('Sorting database compounds by masses')
            compound_mass_list = [Formula(compound.chemical_formula).mass for compound in self.database]
            sort_index = np.argsort(compound_mass_list)
            self.compound_mass_list = np.array(compound_mass_list)[sort_index].tolist()
            self.compound_list = np.array(self.database)[sort_index].tolist()

    def sample(self, mz_range, rt_range, min_ms1_intensity, n_ms1_peaks, ms_levels, alpha=math.inf,
               fixed_mz=False, adduct_proportion_cutoff=0.05, roi_rt_range=None, include_adducts_isotopes=True,
               get_children_method=GET_MS2_BY_PEAKS, adduct_prior_dict=None):
        self.mz_range = mz_range
        self.rt_range = rt_range
        self.min_ms1_intensity = min_ms1_intensity
        self.n_ms1_peaks = n_ms1_peaks
        self.ms_levels = ms_levels
        self.alpha = alpha
        self.fixed_mz = fixed_mz
        self.adduct_proportion_cutoff = adduct_proportion_cutoff
        self.include_adducts_isotopes = include_adducts_isotopes
        self.get_children_method = get_children_method
        self.adduct_prior_dict = adduct_prior_dict

        # set up some counters
        self.crp_samples = [[] for i in range(self.ms_levels)]
        self.crp_index = [[] for i in range(self.ms_levels)]
        self.counts = [[] for i in range(self.ms_levels)]

        # Report error if tries to use spectra to generate MS2+ spectra
        if get_children_method == GET_MS2_BY_SPECTRA and self.ms_levels > 2:
            NotImplementedError("Using spectra to generate MS2+ spectra is not yet implemented")

        # sample from kernel densities
        if self.ms_levels > 2:
            logger.warning(
                "Warning ms_level > 3 not implemented properly yet. Uses scaled ms_level = 2 information for now")
        n_ms1 = self._get_n(1)
        logger.debug("{} chemicals to be created.".format(n_ms1))
        sampled_peaks = self.peak_sampler.get_peak(1, n_ms1, self.mz_range[0][0], self.mz_range[0][1],
                                                   self.rt_range[0][0],
                                                   self.rt_range[0][1], self.min_ms1_intensity)
        # Get formulae from database and check there are enough of them
        self.formula_list = self._sample_formulae(sampled_peaks)

        # Get file split information
        split = self._get_n_ROI_files()

        # create chemicals
        chemicals = []
        # load first ROI file
        current_ROI = 0
        ROIs = self._load_ROI_file(current_ROI, roi_rt_range)
        ROI_intensities = np.array([r.max_intensity for r in ROIs])
        for i in range(n_ms1):
            if i == sum(split[0:(current_ROI + 1)]):
                current_ROI += 1
                ROIs = self._load_ROI_file(current_ROI, roi_rt_range)
                ROI_intensities = np.array([r.max_intensity for r in ROIs])
            formula = self.formula_list[i]
            ROI = ROIs[self._get_ROI_idx(ROI_intensities, sampled_peaks[i].intensity)]
            chem = self._get_known_ms1(formula, ROI, sampled_peaks[i], self.include_adducts_isotopes)
            if self.fixed_mz:
                chem.chromatogram.mzs = [0 for i in range(
                    len(chem.chromatogram.raw_mzs))]
                chem.mzs = [0 for i in range(
                    len(chem.chromatogram.raw_mzs))]
            if ms_levels > 1:
                chem.children = self._get_children(self.get_children_method, chem)
            chem.type = CHEM_DATA
            chemicals.append(chem)
            # if i % 100 == 0:
            #     logger.debug("i = {}".format(i))
        return chemicals

    def _get_n_ROI_files(self):
        count = 0
        for i in range(len(self.ROI_sources)):
            count += len(list(Path(self.ROI_sources[i]).glob('*.p')))
        split = np.array([int(np.floor(self.n_ms1_peaks / count)) for i in range(count)])
        split[0:int(self.n_ms1_peaks - sum(split))] += 1
        return split

    def _load_ROI_file(self, file_index, roi_rt_range=None):
        num_ROI = 0
        for i in range(len(self.ROI_sources)):
            ROI_files = list(Path(self.ROI_sources[i]).glob('*.p'))
            len_ROI = len(ROI_files)
            if len_ROI > file_index:
                ROI_file = ROI_files[file_index - num_ROI]
                ROI = load_obj(ROI_file)
                # logger.debug("Loaded {}".format(ROI_file))
                if roi_rt_range is not None:
                    ROI = self._filter_ROI(ROI, roi_rt_range)
                return ROI
            num_ROI += len_ROI

    def _filter_ROI(self, ROI, roi_rt_range):
        lower = roi_rt_range[0]
        upper = roi_rt_range[1]
        results = [chem for chem in ROI if lower < np.abs(chem.chromatogram.max_rt - chem.chromatogram.min_rt) < upper]
        return results

    def _get_ROI_idx(self, ROI_intensities, intensity):
        return (np.abs(ROI_intensities - intensity)).argmin()

    def _sample_formulae(self, sampled_peaks):
        assert len(sampled_peaks) < len(self.database), 'The number of sampled peaks must be less than ' \
                                                        'the number of database compounds'
        formula_set = set()
        for formula_index in range(len(sampled_peaks)):
            if formula_index % 500 == 0:
                logger.debug('Sampling formula %d/%d' % (formula_index, len(sampled_peaks)))

            mz_peak_sample = sampled_peaks[formula_index].mz
            idx = np.argsort(abs(self.compound_mass_list - mz_peak_sample))

            list_index = 0
            compound_found = False
            while compound_found is False:
                pos = idx[list_index]
                new_compound = self.compound_list[pos].chemical_formula
                if str(new_compound) not in formula_set:
                    formula_set.add(str(new_compound))
                    compound_found = True
                list_index += 1
        return list(formula_set)

    def _get_children(self, get_children_method, parent, n_peaks=None):
        if get_children_method == GET_MS2_BY_SPECTRA:
            kids = self._get_children_spectra(parent)
            return kids
        elif get_children_method == GET_MS2_BY_PEAKS:
            kids = self._get_children_sample(parent, n_peaks)
            return kids
        # TODO: add ability to get children through prediction from parent formula
        # will need to add a default if MS2+ is requested
        else:
            raise ValueError("'get_children_method' must be either 'spectra' or 'sample'")

    def _get_children_spectra(self, parent):
        # spectra is a list containing one MassSpec.Scan object
        found_permissable = False  # reject spectra that comtain no peaks
        while not found_permissable:
            spectra = self.peak_sampler.get_ms2_spectra()[0]
            if len(spectra.intensities) > 0:
                found_permissable = True
        kids = []
        intensity_props = self._get_msn_proportions(None, None, spectra.intensities)
        parent_mass_prop = self.peak_sampler.get_parent_intensity_proportion()
        for i in range(len(spectra.mzs)):
            kid = MSN(spectra.mzs[i], spectra.ms_level, intensity_props[i], parent_mass_prop, None, parent)
            kids.append(kid)
        return kids

    def _get_children_sample(self, parent, n_peaks=None):
        children_ms_level = parent.ms_level + 1
        if n_peaks is None:
            n_peaks = self._get_n(children_ms_level)
        kids = []
        parent_mass_prop = self.peak_sampler.get_parent_intensity_proportion()
        kids_intensity_proportions = self._get_msn_proportions(children_ms_level, n_peaks)
        if self.alpha < math.inf:
            # draws from here if using Chinese Restaurant Process (SLOW!!!)
            for index_children in range(n_peaks):
                next_crp, self.counts[children_ms_level - 1] = Restricted_Crp(self.alpha,
                                                                              self.counts[children_ms_level - 1],
                                                                              self.crp_index[children_ms_level - 1],
                                                                              index_children)
                self.crp_index[children_ms_level - 1].append(next_crp)
                if next_crp == max(self.crp_index[children_ms_level - 1]):
                    kid = self._get_unknown_msn(children_ms_level, parent)
                    kid.prop_ms2_mass = kids_intensity_proportions[index_children]
                    if children_ms_level < self.ms_levels:
                        kid.children = self._get_children(self.get_children_method, kid)
                    self.crp_samples[children_ms_level - 1].append(kid)
                else:
                    kid = copy.deepcopy(self.crp_samples[children_ms_level - 1][next_crp])
                    kid.parent_mass_prop = parent_mass_prop
                    kid.parent = parent
                kids.append(kid)
            self.crp_samples[children_ms_level - 1].extend(kids)
        else:
            # Draws from here if children all independent
            for index_children in range(n_peaks):
                kid = self._get_unknown_msn(children_ms_level, parent)
                kid.prop_ms2_mass = kids_intensity_proportions[index_children]
                kid.parent_mass_prop = parent_mass_prop
                if children_ms_level < self.ms_levels:
                    kid.children = self._get_children(self.get_children_method, kid)
                kids.append(kid)
        return kids

    def _get_msn_proportions(self, children_ms_level=None, n_peaks=None, children_intensities=None):
        if children_intensities is None:
            if children_ms_level == 2:
                kids_intensities = self.peak_sampler.get_peak(children_ms_level, n_peaks)
            else:
                kids_intensities = self.peak_sampler.get_peak(2, n_peaks)
            kids_intensities_total = sum([x.intensity for x in kids_intensities])
            kids_intensities_proportion = [x.intensity / kids_intensities_total for x in kids_intensities]
        else:
            kids_intensities = children_intensities
            kids_intensities_total = sum(kids_intensities)
            kids_intensities_proportion = kids_intensities / kids_intensities_total
        return kids_intensities_proportion

    def _get_n(self, ms_level):
        if ms_level == 1:
            return int(self.n_ms1_peaks)
        elif ms_level == 2:
            return int(self.peak_sampler.n_peaks(2, 1))
        else:
            return int(math.floor(self.peak_sampler.n_peaks(2, 1) / (5 ** (ms_level - 2))))

    def _get_known_ms1(self, formula, ROI, sampled_peak, include_adducts_isotopes):  # fix this
        ## from sampled_peak.rt (XCMS output), we get the point where maximum intensity occurs
        ## so when convering ROI to chemicals, we want to adjust the RT to align it with the point where max intensity occurs
        rt = sampled_peak.rt
        min2mid_rt_ROI = list(ROI.chromatogram.rts[np.where(ROI.chromatogram.intensities == 1)])[0]
        adjusted_rt = rt - min2mid_rt_ROI
        intensity = sampled_peak.intensity
        formula = Formula(formula)
        isotopes = Isotopes(formula)
        adducts = Adducts(formula, self.adduct_proportion_cutoff, adduct_prior_dict=self.adduct_prior_dict)
        return KnownChemical(formula, isotopes, adducts, adjusted_rt, intensity, ROI.chromatogram, None,
                             include_adducts_isotopes)

    def _get_unknown_msn(self, ms_level, parent=None):  # fix this
        if ms_level == 2:
            mz = self.peak_sampler.get_peak(ms_level, 1)[0].mz
        else:
            mz = self.peak_sampler.get_peak(2, 1)[0].mz
        return MSN(mz, ms_level, None, None, None, parent)

    def _valid_ms1_chem(self, chem):
        if chem.max_intensity < self.min_ms1_intensity:
            return False
        elif chem.rt < self.rt_range[0][0]:
            return False
        elif chem.rt > self.rt_range[0][1]:
            return False
        return True


class MultiSampleCreator(object):

    def __init__(self, original_dataset, n_samples, classes, intensity_noise_sd,
                 change_probabilities, change_differences_means, change_differences_sds, dropout_probabilities=None,
                 dropout_numbers=None, experimental_classes=None, experimental_probabilitities=None,
                 experimental_sds=None, save_location=None):
        self.original_dataset = original_dataset
        self.n_samples = n_samples
        self.classes = classes
        self.intensity_noise_sd = intensity_noise_sd
        self.change_probabilities = change_probabilities
        self.change_differences_means = change_differences_means
        self.change_differences_sds = change_differences_sds
        self.dropout_probabilities = dropout_probabilities
        self.dropout_numbers = dropout_numbers
        self.experimental_classes = experimental_classes
        self.experimental_probabilitities = experimental_probabilitities
        self.experimental_sds = experimental_sds
        self.save_location = save_location

        self.sample_classes = []
        for index_classes in range(len(self.classes)):
            self.sample_classes.extend([self.classes[index_classes] for i in range(n_samples[index_classes])])
        self.chemical_statuses = self._get_chemical_statuses()
        self.chemical_differences_from_class1 = self._get_chemical_differences_from_class1()
        if self.experimental_classes is not None:
            self.sample_experimental_statuses = self._get_experimental_statuses()
            self.experimental_effects = self._get_experimental_effects()
        logger.debug("Classes, Statuses and Differences defined.")

        self.samples = []
        for index_sample in range(sum(self.n_samples)):
            logger.debug("Dataset {} of {} created.".format(index_sample + 1, sum(self.n_samples)))
            new_sample = copy.deepcopy(self.original_dataset)
            which_class = np.where(np.array(self.classes) == self.sample_classes[index_sample])
            for index_chemical in range(len(new_sample)):
                if not np.array(self.chemical_statuses)[which_class][0][index_chemical] == "missing":
                    original_intensity = new_sample[index_chemical].max_intensity
                    intensity = self._get_intensity(original_intensity, which_class, index_chemical)
                    adjusted_intensity = self._get_experimental_factor_effect(intensity, index_sample, index_chemical)
                    noisy_adjusted_intensity = self._get_noisy_intensity(adjusted_intensity)
                    new_sample[index_chemical].max_intensity = noisy_adjusted_intensity.tolist()[0]
            chemicals_to_keep = np.where((np.array(self.chemical_statuses)[which_class][0]) != "missing")
            new_sample = np.array(new_sample)[chemicals_to_keep].tolist()
            if self.save_location is not None:
                save_obj(new_sample, Path(self.save_location, 'sample_%d.p' % index_sample))
            self.samples.append(new_sample)

    def _get_chemical_statuses(self):
        chemical_statuses = [np.array(["unchanged" for i in range(len(self.original_dataset))])]
        chemical_statuses.extend([np.random.choice(["changed", "unchanged"], len(self.original_dataset),
                                                   p=[self.change_probabilities[i], 1 - self.change_probabilities[i]])
                                  for i in range(len(self.classes) - 1)])
        self.missing = self._get_missing_chemicals(chemical_statuses)
        self.missing_chemicals = [np.array(self.original_dataset)[miss].tolist() for miss in self.missing]
        for index_chemical in range(len(chemical_statuses)):
            chemical_statuses[index_chemical][self.missing[index_chemical]] = "missing"
        return chemical_statuses

    def _get_missing_chemicals(self, chemical_statuses):
        missing = []
        while len(missing) != len(chemical_statuses):
            if self.dropout_probabilities is not None:
                if self.dropout_numbers is not None:
                    logger.debug("using dropout_probabilties rather than dropout_number.")
                new_missing = list(np.where(np.random.binomial(1, self.dropout_probabilities[len(missing)],
                                                               len(self.original_dataset)))[0])
            if self.dropout_probabilities is None and self.dropout_numbers is not None:
                # new_missing = random.sample(range(0, len(self.original_dataset)), self.dropout_numbers)
                #  CHANGED BY SR, no testing. 
                new_missing = list(np.random.choice(range(0, len(self.original_dataset)), self.dropout_numbers))
            missing.append(new_missing)
            missing = [list(x) for x in set(tuple(sorted(x)) for x in missing)]
        return missing

    def _get_experimental_statuses(self):
        experimental_statuses = []
        for i in range(len(self.experimental_classes)):
            class_allocation = np.random.choice(self.experimental_classes[i], sum(self.n_samples),
                                                p=self.experimental_probabilitities[i])
            experimental_statuses.append(class_allocation)
        return experimental_statuses

    def _get_experimental_effects(self):
        experimental_effects = []
        for i in range(len(self.experimental_classes)):
            coef = [np.random.normal(0, self.experimental_sds[i], len(self.experimental_classes[i])) for j in
                    range(len(self.original_dataset))]
            experimental_effects.append(coef)
        return experimental_effects

    def _get_chemical_differences_from_class1(self):
        chemical_differences_from_class1 = [np.array([0 for i in range(len(self.original_dataset))]) for j in
                                            range(len(self.classes))]
        for index_classes in range(1, len(self.classes)):
            coef_mean = self.change_differences_means[index_classes - 1]
            coef_sd = self.change_differences_sds[index_classes - 1]
            coef_len = sum(self.chemical_statuses[index_classes] == "changed")
            coef = np.random.normal(coef_mean, coef_sd, coef_len)
            chemical_differences_from_class1[index_classes][
                np.where(self.chemical_statuses[index_classes] == "changed")] = coef
        return chemical_differences_from_class1

    def _get_intensity(self, original_intensity, which_class, index_chemical):
        intensity = original_intensity + self.chemical_differences_from_class1[which_class[0][0]][index_chemical]
        return intensity

    def _get_experimental_factor_effect(self, intensity, index_sample, index_chemical):
        experimental_factor_effect = 0.0
        if self.experimental_classes == None:
            return intensity
        else:
            for index_factor in range(len(self.experimental_classes)):
                which_experimental_status = self.sample_experimental_statuses[index_factor][index_sample]
                which_experimental_class = np.where(
                    np.array(self.experimental_classes[index_factor]) == which_experimental_status)
                experimental_factor_effect += self.experimental_effects[index_factor][index_chemical][
                    which_experimental_class]
        return intensity + experimental_factor_effect

    def _get_noisy_intensity(self, adjusted_intensity):
        noisy_intensity = adjusted_intensity + np.random.normal(0, self.intensity_noise_sd[0], 1)
        if noisy_intensity < 0:
            logger.warning("Warning: Negative Intensities have been created")
        return noisy_intensity


class ChemicalMixtureCreator(object):
    # class to create a list of known chemical objects
    # using simplified, cleaned methods
    def __init__(self, formula_sampler,
                 rt_and_intensity_sampler=UniformRTAndIntensitySampler(),
                 chromatogram_sampler=GaussianChromatogramSampler(),
                 ms2_sampler=UniformMS2Sampler(),
                 adduct_proportion_cutoff=0.05,
                 adduct_prior_dict=None):
        self.formula_sampler = formula_sampler
        self.rt_and_intensity_sampler = rt_and_intensity_sampler
        self.chromatogram_sampler = chromatogram_sampler
        self.ms2_sampler = ms2_sampler
        self.adduct_proportion_cutoff = 0.05
        self.adduct_prior_dict = adduct_prior_dict

        # if self.database is not None:
        #     logger.debug('Sorting database compounds by masses')
        #     self.database.sort(key = lambda x: Formula(x.chemical_formula).mass)

    def sample(self, n_chemicals, ms_levels, include_adducts_isotopes=True):

        formula_list = self.formula_sampler.sample(n_chemicals)
        rt_list = []
        intensity_list = []
        chromatogram_list = []
        for formula in formula_list:
            rt, intensity = self.rt_and_intensity_sampler.sample(formula)
            rt_list.append(rt)
            intensity_list.append(intensity)
            chromatogram_list.append(self.chromatogram_sampler.sample(formula, rt, intensity))
        logger.debug('Sampled rt and intensity values and chromatograms')

        # make into known chemical objects
        chemicals = []
        for i, formula in enumerate(formula_list):
            rt = rt_list[i]
            max_intensity = intensity_list[i]
            chromatogram = chromatogram_list[i]
            if isinstance(formula, Formula):
                isotopes = Isotopes(formula)
                adducts = Adducts(formula, self.adduct_proportion_cutoff, adduct_prior_dict=self.adduct_prior_dict)

                chemicals.append(KnownChemical(formula, isotopes, adducts, rt, max_intensity, chromatogram,
                                               include_adducts_isotopes=include_adducts_isotopes))
            elif isinstance(formula, DummyFormula):
                chemicals.append(UnknownChemical(formula.mass, rt, max_intensity, chromatogram))
            else:
                logger.warning("Unkwown formula object: {}".format(type(formula)))

            if ms_levels == 2:
                parent = chemicals[-1]
                child_mz, child_intensity, parent_proportion = self.ms2_sampler.sample(formula)

                children = []
                for mz, intensity in zip(child_mz, child_intensity):
                    child = MSN(mz, 2, intensity, parent_proportion, None, parent)
                    children.append(child)
                children.sort(key=lambda x: x.isotopes[0])
                parent.children = children

        return chemicals


class MultipleMixtureCreator(object):
    def __init__(self, master_chemical_list, group_list, group_dict,
                 intensity_noise=GaussianPeakNoise(sigma=0.001, log_space=True), overall_missing_probability=0.0):
        self.master_chemical_list = master_chemical_list
        self.group_list = group_list
        self.group_dict = group_dict
        self.intensity_noise = intensity_noise
        self.overall_missing_probability = overall_missing_probability

        if not 'control' in self.group_dict:
            self.group_dict['control'] = {}
            self.group_dict['control']['missing_probability'] = 0.0
            self.group_dict['control']['changing_probability'] = 0.0

        self._generate_changes()

    def _generate_changes(self):
        self.group_multipliers = {}
        for group in self.group_dict:
            self.group_multipliers[group] = {}
            missing_probability = self.group_dict[group]['missing_probability']
            changing_probability = self.group_dict[group]['changing_probability']
            for chemical in self.master_chemical_list:
                self.group_multipliers[group][chemical] = 1.0  # default is no change
                if np.random.rand() <= changing_probability:
                    self.group_multipliers[group][chemical] = np.exp(np.random.rand() * (
                                np.log(5) - np.log(0.2) + np.log(0.2)))  # uniform between doubling and halving
                if np.random.rand() <= missing_probability:
                    self.group_multipliers[group][chemical] = 0.0

    def generate_chemical_lists(self):
        chemical_lists = []
        for group in self.group_list:
            new_list = []
            for chemical in self.master_chemical_list:
                if np.random.rand() < self.overall_missing_probability or self.group_multipliers[group][chemical] == 0.:
                    continue  # chemical is missing overall
                new_intensity = chemical.max_intensity * self.group_multipliers[group][chemical]
                new_intensity = self.intensity_noise.get(new_intensity, 1)

                # make a new known chemical
                new_chemical = copy.deepcopy(chemical)
                new_chemical.max_intensity = new_intensity
                new_chemical.original_chemical = chemical
                new_list.append(new_chemical)
            chemical_lists.append(new_list)
        return chemical_lists
