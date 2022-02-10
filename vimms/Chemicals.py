"""
Provides implementation of Chemicals objects that are used as input
to the simulation.
"""
import copy

import numpy as np
import scipy
import scipy.stats
from loguru import logger

from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, \
    GaussianChromatogramSampler, UniformMS2Sampler
from vimms.Chromatograms import EmpiricalChromatogram
from vimms.Common import POS_TRANSFORMATIONS, Formula, DummyFormula, \
    PROTON_MASS, POSITIVE, NEGATIVE, C12_PROPORTION, \
    C13_MZ_DIFF, C, MONO, C13
from vimms.Noise import GaussianPeakNoise
from vimms.Roi import make_roi, RoiParams


class DatabaseCompound(object):
    """
    A class to represent a compound stored in a database, e.g. HMDB
    """

    def __init__(self, name, chemical_formula, monisotopic_molecular_weight,
                 smiles, inchi, inchikey):
        self.name = name
        self.chemical_formula = chemical_formula
        self.monisotopic_molecular_weight = monisotopic_molecular_weight
        self.smiles = smiles
        self.inchi = inchi
        self.inchikey = inchikey


class Isotopes(object):
    """
    A class to represent an isotope of a chemical
    """

    def __init__(self, formula):
        self.formula = formula

    def get_isotopes(self, total_proportion):
        """
        Gets the isotopes
        TODO: Add functionality for elements other than Carbon
        """
        peaks = [() for i in
                 range(len(self._get_isotope_proportions(total_proportion)))]
        for i in range(len(peaks)):
            peaks[i] += (self._get_isotope_mz(self._get_isotope_names(i)),)
            peaks[i] += (self._get_isotope_proportions(total_proportion)[i],)
            peaks[i] += (self._get_isotope_names(i),)
        return peaks

    def _get_isotope_proportions(self, total_proportion):
        """
        Get isotope proportion by sampling from a binomial pmf
        """
        proportions = []
        while sum(proportions) < total_proportion:
            proportions.extend(
                [scipy.stats.binom.pmf(len(proportions),
                                       self.formula._get_n_element(C),
                                       1 - C12_PROPORTION)])
        normalised_proportions = [proportions[i] / sum(proportions) for i in
                                  range(len(proportions))]
        return normalised_proportions

    def _get_isotope_names(self, isotope_number):
        if isotope_number == 0:
            return MONO
        else:
            return str(isotope_number) + C13

    def _get_isotope_mz(self, isotope):
        if isotope == MONO:
            return self.formula._get_mz()
        elif isotope[-3:] == C13:
            return self.formula._get_mz() + float(
                isotope.split(C13)[0]) * C13_MZ_DIFF
        else:
            return None


class Adducts(object):
    """
    A class to represent an adduct of a chemical
    """

    def __init__(self, formula, adduct_proportion_cutoff=0.05,
                 adduct_prior_dict=None):
        if adduct_prior_dict is None:
            self.adduct_names = {POSITIVE: list(POS_TRANSFORMATIONS.keys())}
            self.adduct_prior = {
                POSITIVE: np.ones(len(self.adduct_names[POSITIVE])) * 0.1}
            self.adduct_prior[POSITIVE][
                0] = 1.0  # give more weight to the first one, i.e. M+H
        else:
            assert POSITIVE in adduct_prior_dict or \
                   NEGATIVE in adduct_prior_dict
            self.adduct_names = {k: list(adduct_prior_dict[k].keys()) for k in
                                 adduct_prior_dict}
            self.adduct_prior = {
                k: np.array(list(adduct_prior_dict[k].values())) for k in
                adduct_prior_dict}
        self.formula = formula
        self.adduct_proportion_cutoff = adduct_proportion_cutoff

    def get_adducts(self):
        """
        Get the adducts
        """
        adducts = {}
        proportions = self._get_adduct_proportions()
        for k in self.adduct_names:
            adducts[k] = []
            for j in range(len(self.adduct_names[k])):
                if proportions[k][j] != 0:
                    adducts[k].extend(
                        [(self._get_adduct_names()[k][j], proportions[k][j])])
        return adducts

    def _get_adduct_proportions(self):
        """
        Get adducts according to a dirichlet distribution
        """
        # TODO: replace this with something proper
        proportions = {}
        for k in self.adduct_prior:
            proportions[k] = np.random.dirichlet(self.adduct_prior[k])
            while max(proportions[k]) < 0.2:
                proportions[k] = np.random.dirichlet(self.adduct_prior[k])
            proportions[k][
                np.where(proportions[k] < self.adduct_proportion_cutoff)] = 0
            proportions[k] = proportions[k] / max(proportions[k])
            proportions[k].tolist()
            assert len(proportions[k]) == len(self.adduct_names[k])
        return proportions

    def _get_adduct_names(self):
        return self.adduct_names


class Chemical(object):
    """
    The base class that represents a Chemical object.
    Should be realised as either Known or Unknown chemicals.
    """

    def __repr__(self):
        raise NotImplementedError()


class UnknownChemical(Chemical):
    """
    A Chemical representation from an unknown chemical formula
    """

    def __init__(self, mz, rt, max_intensity, chromatogram, children=None,
                 base_chemical=None):
        self.max_intensity = max_intensity
        self.isotopes = [
            (mz, 1, "Mono")]  # [(mz, intensity_proportion, isotope,name)]
        self.adducts = {POSITIVE: [("M+H", 1)], NEGATIVE: [("M-H", 1)]}
        self.rt = rt
        self.chromatogram = chromatogram
        self.children = children
        self.ms_level = 1
        self.mz_diff = 0
        self.mass = mz
        self.base_chemical = base_chemical

    def __repr__(self):
        return 'UnknownChemical mz=%.4f rt=%.2f max_intensity=%.2f' % (
            self.isotopes[0][0], self.rt, self.max_intensity)

    def get_key(self):
        """
        Turns a chemical object into (mz, rt, intensity) tuples for
        equal comparison
        :return: a tuple of the three values
        """
        return (tuple(self.isotopes), self.rt, self.max_intensity)

    def __eq__(self, other):
        if not isinstance(other, UnknownChemical):
            return False
        return self.get_key() == other.get_key()

    def __hash__(self):
        return hash(self.get_key())

    def get_apex_rt(self):
        return self.rt + self.chromatogram.get_apex_rt()

    def get_original_parent(self):
        return self if self.base_chemical is None else \
            self.base_chemical.get_original_parent()


class KnownChemical(Chemical):
    """
    A Chemical representation from a known chemical formula
    """

    def __init__(self, formula, isotopes, adducts, rt, max_intensity,
                 chromatogram, children=None,
                 include_adducts_isotopes=True, total_proportion=0.99,
                 database_accession=None, base_chemical=None):
        self.formula = formula
        self.mz_diff = C13_MZ_DIFF
        if include_adducts_isotopes is True:
            self.isotopes = isotopes.get_isotopes(total_proportion)
            self.adducts = adducts.get_adducts()
        else:
            mz = isotopes.get_isotopes(total_proportion)[0][0]
            self.isotopes = [(mz, 1, MONO)]
            self.adducts = {POSITIVE: [("M+H", 1)], NEGATIVE: [("M-H", 1)]}
        self.rt = rt
        self.max_intensity = max_intensity
        self.chromatogram = chromatogram
        self.children = children
        self.ms_level = 1
        self.mass = self.formula.mass
        self.database_accession = database_accession
        self.base_chemical = base_chemical

    def __repr__(self):
        return 'KnownChemical - %r rt=%.2f max_intensity=%.2f' % (
            self.formula.formula_string, self.rt, self.max_intensity)

    def get_key(self):
        return (tuple(self.formula.formula_string), self.rt)

    def __eq__(self, other):
        if not isinstance(other, KnownChemical):
            return False
        return self.get_key() == other.get_key()

    def __hash__(self):
        return hash(self.get_key())

    def get_apex_rt(self):
        return self.rt + self.chromatogram.get_apex_rt()

    def get_original_parent(self):
        return self if self.base_chemical is None else \
            self.base_chemical.get_original_parent()


class MSN(Chemical):
    """
    A chemical that represents an MS2+ fragment.
    """

    def __init__(self, mz, ms_level, prop_ms2_mass, parent_mass_prop,
                 children=None, parent=None):
        self.isotopes = [(mz, None, "MSN")]
        self.ms_level = ms_level
        self.prop_ms2_mass = prop_ms2_mass
        self.parent_mass_prop = parent_mass_prop
        self.children = children
        self.parent = parent

    def __repr__(self):
        return 'MSN Fragment mz=%.4f ms_level=%d' % (
            self.isotopes[0][0], self.ms_level)


class ChemicalMixtureCreator(object):
    '''
    A class to create a list of known chemical objects using simplified,
    cleaned methods.
    '''

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
        #     self.database.sort(
        #         key = lambda x: Formula(x.chemical_formula).mass)

    def sample(self, n_chemicals, ms_levels, include_adducts_isotopes=True):
        '''
        Samples chemicals.
        '''

        formula_list = self.formula_sampler.sample(n_chemicals)
        rt_list = []
        intensity_list = []
        chromatogram_list = []
        for formula, db_accession in formula_list:
            rt, intensity = self.rt_and_intensity_sampler.sample(formula)
            rt_list.append(rt)
            intensity_list.append(intensity)
            chromatogram_list.append(
                self.chromatogram_sampler.sample(formula, rt, intensity))
        logger.debug('Sampled rt and intensity values and chromatograms')

        # make into known chemical objects
        chemicals = []
        for i, (formula, db_accession) in enumerate(formula_list):
            rt = rt_list[i]
            max_intensity = intensity_list[i]
            chromatogram = chromatogram_list[i]
            if isinstance(formula, Formula):
                isotopes = Isotopes(formula)
                adducts = Adducts(formula, self.adduct_proportion_cutoff,
                                  adduct_prior_dict=self.adduct_prior_dict)

                chemicals.append(
                    KnownChemical(
                        formula, isotopes, adducts, rt, max_intensity,
                        chromatogram,
                        include_adducts_isotopes=include_adducts_isotopes,
                        database_accession=db_accession))
            elif isinstance(formula, DummyFormula):
                chemicals.append(
                    UnknownChemical(formula.mass, rt, max_intensity,
                                    chromatogram))
            else:
                logger.warning(
                    "Unkwown formula object: {}".format(type(formula)))

            if ms_levels == 2:
                parent = chemicals[-1]
                child_mz, child_intensity, parent_proportion = \
                    self.ms2_sampler.sample(parent)

                children = []
                for mz, intensity in zip(child_mz, child_intensity):
                    child = MSN(mz, 2, intensity, parent_proportion, None,
                                parent)
                    children.append(child)
                children.sort(key=lambda x: x.isotopes[0])
                parent.children = children

        return chemicals


class MultipleMixtureCreator(object):
    '''
    A class to create a list of known chemical objects in multiple
    samples (mixtures)
    '''

    def __init__(self, master_chemical_list, group_list, group_dict,
                 intensity_noise=GaussianPeakNoise(
                     sigma=0.001, log_space=True),
                 overall_missing_probability=0.0):
        # example
        # group_list = ['control', 'control', 'case', 'case']
        # group_dict = {
        #     'control': {
        #         'missing_probability': 0.0,
        #         'changing_probability': 0.0
        #     }, 'case': {
        #         'missing_probability': 0.0,
        #         'changing_probability': 0.0
        #     }
        # }
        self.master_chemical_list = master_chemical_list
        self.group_list = group_list
        self.group_dict = group_dict
        self.intensity_noise = intensity_noise
        self.overall_missing_probability = overall_missing_probability

        if 'control' not in self.group_dict:
            self.group_dict['control'] = {}
            self.group_dict['control']['missing_probability'] = 0.0
            self.group_dict['control']['changing_probability'] = 0.0

        self._generate_changes()

    def _generate_changes(self):
        self.group_multipliers = {}
        for group in self.group_dict:
            self.group_multipliers[group] = {}
            missing_probability = self.group_dict[group]['missing_probability']
            changing_probability = self.group_dict[group][
                'changing_probability']
            for chemical in self.master_chemical_list:
                self.group_multipliers[group][
                    chemical] = 1.0  # default is no change
                if np.random.rand() <= changing_probability:
                    self.group_multipliers[group][chemical] = np.exp(
                        np.random.rand() * (
                            np.log(5) - np.log(0.2) + np.log(
                                0.2)))  # uniform between doubling and halving
                if np.random.rand() <= missing_probability:
                    self.group_multipliers[group][chemical] = 0.0

    def generate_chemical_lists(self):
        chemical_lists = []
        for group in self.group_list:
            new_list = []
            for chemical in self.master_chemical_list:
                if np.random.rand() < self.overall_missing_probability or \
                        self.group_multipliers[group][chemical] == 0.:
                    continue  # chemical is missing overall
                new_intensity = chemical.max_intensity * \
                    self.group_multipliers[group][chemical]
                new_intensity = self.intensity_noise.get(new_intensity, 1)

                # make a new known chemical
                new_chemical = copy.deepcopy(chemical)
                new_chemical.max_intensity = new_intensity
                new_chemical.base_chemical = chemical
                new_list.append(new_chemical)
            chemical_lists.append(new_list)
        return chemical_lists


class ChemicalMixtureFromMZML(object):
    '''
    A class to create a list of known chemical objects from an mzML file
    using simplified, cleaned methods.
    '''

    def __init__(self, mzml_file_name, ms2_sampler=UniformMS2Sampler(),
                 roi_params=None):
        self.mzml_file_name = mzml_file_name
        self.ms2_sampler = ms2_sampler
        self.roi_params = roi_params

        if roi_params is None:
            self.roi_params = RoiParams()

        self.good_rois = self._extract_rois()
        assert len(self.good_rois) > 0

    def _extract_rois(self):
        good = make_roi(str(self.mzml_file_name), self.roi_params)
        logger.debug("Extracted {} good ROIs from {}".format(
            len(good), self.mzml_file_name))
        return good

    def sample(self, n_chemicals, ms_levels, source_polarity=POSITIVE):
        """
            Generate a dataset from the mzml file
            n_chemicals: set to None if you want all the ROIs turned
            into chemicals
        """
        if n_chemicals is None:
            rois_to_use = range(len(self.good_rois))
        elif n_chemicals > len(self.good_rois):
            rois_to_use = range(len(self.good_rois))
            logger.warning("Requested more chemicals than ROIs")
        else:
            rois_to_use = np.random.permutation(len(self.good_rois))[
                :n_chemicals]
        chemicals = []
        for roi_idx in rois_to_use:
            r = self.good_rois[roi_idx]
            mz = r.get_mean_mz()
            if source_polarity == POSITIVE:
                mz -= PROTON_MASS
            elif source_polarity == NEGATIVE:
                mz += PROTON_MASS
            else:
                logger.warning(
                    "Unknown source polarity {}".format(source_polarity))
            rt = r.rt_list[0]  # this is in seconds
            max_intensity = max(r.intensity_list)

            #  make a chromatogram object
            chromatogram = EmpiricalChromatogram(np.array(r.rt_list),
                                                 np.array(r.mz_list),
                                                 np.array(r.intensity_list),
                                                 single_point_length=0.9)

            # make a chemical
            new_chemical = UnknownChemical(mz, rt, max_intensity, chromatogram,
                                           children=None)
            chemicals.append(new_chemical)

            if ms_levels == 2:
                parent = chemicals[-1]
                child_mz, child_intensity, parent_proportion = \
                    self.ms2_sampler.sample(parent)

                children = []
                for mz, intensity in zip(child_mz, child_intensity):
                    child = MSN(mz, 2, intensity, parent_proportion, None,
                                parent)
                    children.append(child)
                children.sort(key=lambda x: x.isotopes[0])
                parent.children = children

        return chemicals


def get_pooled_sample(dataset_list):
    '''
    Takes a list of datasets and creates a pooled dataset from them
    '''
    n_datasets = len(dataset_list)
    all_chems = np.array(
        [item for sublist in dataset_list for item in sublist])
    unique_parents = list(set([chem.base_chemical for chem in all_chems]))
    # create dataset
    dataset = []
    for chem in unique_parents:
        matched_chemicals = all_chems[np.where(all_chems == chem)[0]]
        new_intensity = sum(
            [mchem.max_intensity for mchem in matched_chemicals]) / n_datasets
        new_chem = copy.deepcopy(chem)
        new_chem.max_intensity = new_intensity
        dataset.append(new_chem)
    return dataset
