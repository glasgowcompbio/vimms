"""
Provides implementation of Chemicals objects that are used as input
to the simulation.
"""
import copy
from abc import ABC, abstractmethod

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
from vimms.Roi import make_roi, RoiBuilderParams


class DatabaseCompound():
    """
    A class to represent a compound stored in a database, e.g. HMDB
    """

    def __init__(self, name, chemical_formula, monisotopic_molecular_weight,
                 smiles, inchi, inchikey):
        """
        Creates a DatabaseCompound object
        Args:
            name: the compound name
            chemical_formula: the formula of that compound
            monisotopic_molecular_weight: the monoisotopic weight of the compound
            smiles: SMILES of the compound
            inchi: InCHI of the compound
            inchikey: InCHI key of the compound
        """
        self.name = name
        self.chemical_formula = chemical_formula
        self.monisotopic_molecular_weight = monisotopic_molecular_weight
        self.smiles = smiles
        self.inchi = inchi
        self.inchikey = inchikey


class Isotopes():
    """
    A class to represent an isotope of a chemical
    """

    def __init__(self, formula):
        """
        Create an Isotope object
        Args:
            formula: the formula for the given isotope
        """
        self.formula = formula

    def get_isotopes(self, total_proportion):
        """
        Gets the isotope total proportion

        Args:
            total_proportion: the total proportion to compute

        Returns: the computed isotope total proportion

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

        Args:
            total_proportion: the total proportion to compute

        Returns: the computed isotope total proportion

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
        """
        Get the isotope name given the number, e.g. 0 is the monoisotope
        Args:
            isotope_number: the isotope number

        Returns: the isotope name

        """
        if isotope_number == 0:
            return MONO
        else:
            return str(isotope_number) + C13

    def _get_isotope_mz(self, isotope):
        """
        Get the isotope m/z value
        Args:
            isotope: the isotope name

        Returns: the isotope m/z value

        """
        if isotope == MONO:
            return self.formula._get_mz()
        elif isotope[-3:] == C13:
            return self.formula._get_mz() + float(
                isotope.split(C13)[0]) * C13_MZ_DIFF
        else:
            return None


class Adducts():
    """
    A class to represent an adduct of a chemical
    """

    def __init__(self, formula, adduct_proportion_cutoff=0.05,
                 adduct_prior_dict=None):
        """
        Create an Adduct class

        Args:
            formula: the formula of this adduct
            adduct_proportion_cutoff: proportion cut-off of the adduct
            adduct_prior_dict: custom adduct dictionary, if any
        """
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
        Returns: adducts in the correct proportion
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

        Returns: adduct proportion after sampling

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
        """
        Get the adduct name
        Returns: adduct name

        """
        return self.adduct_names


class BaseChemical(ABC):
    """
    The base class for Chemical objects across all MS levels.
    Chemicals at MS level = 1 is special and should be instantiated as either Known
    or Unknown chemicals.
    For other MS levels, please use the MSN class.
    """

    def __init__(self, ms_level, children):
        """
        Defines a base chemical object
        Args:
            ms_level: the MS level of this chemical
            children: any children of this chemical
        """
        self.ms_level = ms_level
        self.children = children


class Chemical(BaseChemical):
    """
    The class that represents a Chemical object of MS-level 1.
    Should be realised as either Known or Unknown chemicals.
    """

    def __init__(self, rt, max_intensity, chromatogram, children,
                 base_chemical):
        """
        Create a Chemical object
        Args:
            rt: the RT value of this chemical
            max_intensity: the maximum intensity of this chemical
            chromatogram: the chromatogram of this chemical
            children: any children of this chemical
            base_chemical: the base chemical from which this chemical is derived
        """
        ms_level = 1
        super().__init__(ms_level, children)

        self.rt = rt
        self.max_intensity = max_intensity
        self.chromatogram = chromatogram
        self.mz_diff = 0
        self.base_chemical = base_chemical

    def get_apex_rt(self):
        """
        Get the apex (highest point) RT of the chromatogram of this chemical
        Returns: the apex RT of the chromatogram

        """

        return self.rt + self.chromatogram.get_apex_rt()

    def get_original_parent(self):
        """
        Get the original base chemical in a recursive manner.
        This is necessary if the parent chemical also has another parent.
        Returns: the original base chemical

        """
        return self if self.base_chemical is None else \
            self.base_chemical.get_original_parent()

    @abstractmethod
    def get_key(self):
        """
        Turns a chemical object into some sensible keys for comparison

        Returns: keys for comparisons in __eq__ and __hash__

        """
        pass

    def __eq__(self, other):
        if type(other) is type(self):
            return self.get_key() == other.get_key()
        return False

    def __hash__(self):
        return hash(self.get_key())


class UnknownChemical(Chemical):
    """
    A Chemical representation from an unknown chemical formula.
    Unknown chemicals are typically created by extracting Regions-of-Interest
    from an existing mzML file.
    """

    def __init__(self, mz, rt, max_intensity, chromatogram, children=None,
                 base_chemical=None):
        """
        Initialises an UnknownChemical object.

        Args:
            mz: the m/z value of this chemical. Unlike [vimms.Chemicals.KnownChemical][] here we
                know the m/z value but do not known the formula that generates this chemical.
            rt: the RT value of this chemical
            max_intensity: the maximum intensity of this chemical
            chromatogram: the chromatogram of this chemical
            children: any children of this chemical
            base_chemical: the base chemical from which this chemical is derived
        """
        super().__init__(rt, max_intensity, chromatogram, children, base_chemical)
        self.isotopes = [
            (mz, 1, "Mono")]  # [(mz, intensity_proportion, isotope,name)]
        self.adducts = {POSITIVE: [("M+H", 1)], NEGATIVE: [("M-H", 1)]}
        self.mass = mz

    def get_key(self):
        return tuple(self.isotopes), self.rt, self.max_intensity

    def __repr__(self):
        return 'UnknownChemical mz=%.4f rt=%.2f max_intensity=%.2f' % (
            self.isotopes[0][0], self.rt, self.max_intensity)


class KnownChemical(Chemical):
    """
    A Chemical representation from a known chemical formula.
    Known chemicals have formula which are defined during creation.
    """

    def __init__(self, formula, isotopes, adducts, rt, max_intensity,
                 chromatogram, children=None,
                 include_adducts_isotopes=True, total_proportion=0.99,
                 database_accession=None, base_chemical=None):
        """
        Initialises a Known chemical object

        Args:
            formula: the formula of this chemical object.
            isotopes: the isotope of this chemical object
            adducts: the adduct of this chemical object
            rt: the retention time value of this chemical object
            max_intensity: the maximum intensity value in the chromatogram
            chromatogram: the chromatogram of the chemical
            children: any children of the chemical
            include_adducts_isotopes: whether to include adducts and isotopes of this chemical
            total_proportion: total proportion of this chemical
            database_accession: database accession number, if any
            base_chemical: parent chemica, if any
        """
        super().__init__(rt, max_intensity, chromatogram, children, base_chemical)
        self.formula = formula
        self.mz_diff = C13_MZ_DIFF
        if include_adducts_isotopes is True:
            self.isotopes = isotopes.get_isotopes(total_proportion)
            self.adducts = adducts.get_adducts()
        else:
            mz = isotopes.get_isotopes(total_proportion)[0][0]
            self.isotopes = [(mz, 1, MONO)]
            self.adducts = {POSITIVE: [("M+H", 1)], NEGATIVE: [("M-H", 1)]}
        self.mass = self.formula.mass
        self.database_accession = database_accession

    def get_key(self):
        return tuple(self.formula.formula_string), self.rt

    def __repr__(self):
        return 'KnownChemical - %r rt=%.2f max_intensity=%.2f' % (
            self.formula.formula_string, self.rt, self.max_intensity)


class MSN(BaseChemical):
    """
    A chemical that represents an MS2+ fragment.
    """

    def __init__(self, mz, ms_level, prop_ms2_mass, parent_mass_prop,
                 children=None, parent=None):
        """
        Initialises an MSN object

        Args:
            mz: the m/z value of this fragment peak
            ms_level: the MS level of this fragment peak
            prop_ms2_mass: proportion of MS2 mass
            parent_mass_prop: proportion from the parent MS1 mass
            children: any children
            parent: parent MS1 peak
        """
        super().__init__(ms_level, children)

        self.isotopes = [(mz, None, "MSN")]
        self.ms_level = ms_level
        self.prop_ms2_mass = prop_ms2_mass
        self.parent_mass_prop = parent_mass_prop
        self.parent = parent

    def __repr__(self):
        return 'MSN Fragment mz=%.4f ms_level=%d' % (
            self.isotopes[0][0], self.ms_level)


class ChemicalMixtureCreator():
    """
    A class to create a list of known chemical objects using simplified,
    cleaned methods.
    """

    def __init__(self, formula_sampler,
                 rt_and_intensity_sampler=UniformRTAndIntensitySampler(),
                 chromatogram_sampler=GaussianChromatogramSampler(),
                 ms2_sampler=UniformMS2Sampler(),
                 adduct_proportion_cutoff=0.05,
                 adduct_prior_dict=None):
        """
        Create a mixture of [vimms.Chemicals.KnownChemical][] objects.
        Args:
            formula_sampler: an instance of [vimms.ChemicalSamplers.FormulaSampler][] to sample
                             chemical formulae.
            rt_and_intensity_sampler: an instance of
                                      [vimms.ChemicalSamplers.RTAndIntensitySampler][] to sample
                                      RT and intensity values.
            chromatogram_sampler: an instance of
                                  [vimms.ChemicalSamplers.ChromatogramSampler][] to sample
                                  chromatograms.
            ms2_sampler: an instance of
                         [vimms.ChemicalSamplers.MS2Sampler][] to sample MS2
                         fragmentation spectra.
            adduct_proportion_cutoff: proportion of adduct cut-off
            adduct_prior_dict: custom adduct dictionary
        """
        self.formula_sampler = formula_sampler
        self.rt_and_intensity_sampler = rt_and_intensity_sampler
        self.chromatogram_sampler = chromatogram_sampler
        self.ms2_sampler = ms2_sampler
        self.adduct_proportion_cutoff = adduct_proportion_cutoff
        self.adduct_prior_dict = adduct_prior_dict

        # if self.database is not None:
        #     logger.debug('Sorting database compounds by masses')
        #     self.database.sort(
        #         key = lambda x: Formula(x.chemical_formula).mass)

    def sample(self, n_chemicals, ms_levels, include_adducts_isotopes=True):
        """
        Samples chemicals.

        Args:
            n_chemicals: the number of chemicals
            ms_levels: the highest MS level to generate. Typically this is 2.
            include_adducts_isotopes: whether to include adduct and isotopes or not.

        Returns: a list of [vimms.Chemicals.KnownChemical][] objects.

        """

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


class MultipleMixtureCreator():
    """
    A class to create a list of known chemical objects in multiple
    samples (mixtures)
    """

    def __init__(self, master_chemical_list, group_list, group_dict,
                 intensity_noise=GaussianPeakNoise(
                     sigma=0.001, log_space=True),
                 overall_missing_probability=0.0):
        """
        Create a chemical mixture creator.
        example

        Args:
            master_chemical_list: the master list of Chemicals to create each sample (mixture)
            group_list: a list of different groups, e.g.
                        group_list = ['control', 'control', 'case', 'case']
            group_dict: a dictionary of parameters for each group, e.g.
                        group_dict = {
                            'control': {
                                'missing_probability': 0.0,
                                'changing_probability': 0.0
                            }, 'case': {
                                'missing_probability': 0.0,
                                'changing_probability': 0.0
                            }
                        }
            intensity_noise: intensity noise. Should be an instance of [vimms.Noise.NoPeakNoise][].
            overall_missing_probability: overall missing probability across all mixtures.
        """
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
        """
        Computes changes across groups.
        Returns: None

        """
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
                    # uniform between doubling and halving
                    self.group_multipliers[group][chemical] = np.exp(
                        np.random.rand() * (np.log(5) - np.log(0.2) + np.log(0.2)))
                if np.random.rand() <= missing_probability:
                    self.group_multipliers[group][chemical] = 0.0

    def generate_chemical_lists(self):
        """
        Generates list of chemicals across mixtures (samples)

        Returns: the list of chemicals across mixtures (samples)

        """
        chemical_lists = []
        for group in self.group_list:
            new_list = []
            for chemical in self.master_chemical_list:
                if np.random.rand() < self.overall_missing_probability or \
                        self.group_multipliers[group][chemical] == 0.:
                    continue  # chemical is missing overall
                new_intensity = chemical.max_intensity * self.group_multipliers[group][chemical]
                new_intensity = self.intensity_noise.get(new_intensity, 1)

                # make a new known chemical
                new_chemical = copy.deepcopy(chemical)
                new_chemical.max_intensity = new_intensity
                new_chemical.base_chemical = chemical
                new_list.append(new_chemical)
            chemical_lists.append(new_list)
        return chemical_lists


class ChemicalMixtureFromMZML():
    """
    A class to create a list of known chemical objects from an mzML file
    using simplified, cleaned methods.
    """

    def __init__(self, mzml_file_name, ms2_sampler=UniformMS2Sampler(),
                 roi_params=None):
        """
        Create a ChemicalMixtureFromMZML class.
        Args:
            mzml_file_name: the mzML filename to extract [vimms.Chemicals.UnknownChemical][]
                            objects from.
            ms2_sampler: the MS2 sampler to use. Should be an instance of
                         [vimms.ChemicalSamplers.MS2Sampler][].
            roi_params: parameters for ROI building, as defined in [vimms.Roi.RoiBuilderParams][].
        """
        self.mzml_file_name = mzml_file_name
        self.ms2_sampler = ms2_sampler
        self.roi_params = roi_params

        if roi_params is None:
            self.roi_params = RoiBuilderParams()

        self.good_rois = self._extract_rois()
        assert len(self.good_rois) > 0

    def _extract_rois(self):
        """
        Extract good ROIs from the mzML file.
        Good ROI are ROIs that have been filtered according to certain criteria.

        Returns: the list of good ROI objects
        """
        good = make_roi(str(self.mzml_file_name), self.roi_params)
        logger.debug("Extracted {} good ROIs from {}".format(
            len(good), self.mzml_file_name))
        return good

    def sample(self, n_chemicals, ms_levels, source_polarity=POSITIVE):
        """
        Generate a dataset of Chemicals from the mzml file
        Args:
            n_chemicals: the number of Chemical objects. Set to None to get all the ROIs.
            ms_levels: the maximum MS level
            source_polarity: either POSITIVE or NEGATIVE

        Returns: the list of Chemicals from the mzML file.

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

            # make a chromatogram object
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
    """
    Takes a list of datasets and creates a pooled dataset from them

    Args:
        dataset_list: a list of datasets, each containing Chemical objects

    Returns: combined list where the datasets have been pooled

    """
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
