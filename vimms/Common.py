import collections
import gzip
import logging
import math
import os
import pathlib
import pickle
import re
import sys
import zipfile
from bisect import bisect_left

import numpy as np
import requests
from loguru import logger
from tqdm.auto import tqdm

from mass_spec_utils.data_import.mzml import MZMLFile
###############################################################################
# Common constants
###############################################################################


MONO = 'Mono'
C = 'C'
C13 = 'C13'
C12_PROPORTION = 0.989
C13_MZ_DIFF = 1.0033548378

MZ = 'mz'
INTENSITY = 'intensity'
RT = 'rt'
MZ_INTENSITY_RT = MZ + '_' + INTENSITY + '_' + RT
N_PEAKS = 'n_peaks'
SCAN_DURATION = 'scan_duration'
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
DEFAULT_MS1_SCAN_WINDOW = (70.0, 1000.0)
DEFAULT_MSN_SCAN_WINDOW = (70.0, 600.0)
DEFAULT_ISOLATION_WIDTH = 0.7
CHEM_DATA = 'data'
CHEM_NOISE = 'noise'

MAX_POSSIBLE_RT = 1e6

DEFAULT_MS1_COLLISION_ENERGY = 0
DEFAULT_MS2_COLLISION_ENERGY = 25
DEFAULT_MS1_ORBITRAP_RESOLUTION = 120000
DEFAULT_MS2_ORBITRAP_RESOLUTION = 7500
DEFAULT_MS1_AGC_TARGET = 200000
DEFAULT_MS2_AGC_TARGET = 30000
DEFAULT_MS1_MAXIT = 250
DEFAULT_MS2_MAXIT = 100
DEFAULT_MS1_ACTIVATION_TYPE = 'HCD'  # CID, HCD
DEFAULT_MS2_ACTIVATION_TYPE = 'HCD'  # CID, HCD
DEFAULT_MS1_MASS_ANALYSER = 'Orbitrap'  # IonTrap, Orbitrap
DEFAULT_MS2_MASS_ANALYSER = 'Orbitrap'  # IonTrap, Orbitrap
DEFAULT_MS1_ISOLATION_MODE = 'Quadrupole'  # None, Quadrupole, IonTrap
DEFAULT_MS2_ISOLATION_MODE = 'Quadrupole'  # None, Quadrupole, IonTrap
DEFAULT_SOURCE_CID_ENERGY = 0

PROTON_MASS = 1.00727645199076

# Note: M+H should come first in this dict because of the prior specification
POS_TRANSFORMATIONS = collections.OrderedDict()
POS_TRANSFORMATIONS['M+H'] = lambda mz: (mz + PROTON_MASS)
POS_TRANSFORMATIONS['[M+ACN]+H'] = lambda mz: (mz + 42.033823)
POS_TRANSFORMATIONS['[M+CH3OH]+H'] = lambda mz: (mz + 33.033489)
POS_TRANSFORMATIONS['[M+NH3]+H'] = lambda mz: (mz + 18.033823)
POS_TRANSFORMATIONS['M+Na'] = lambda mz: (mz + 22.989218)
POS_TRANSFORMATIONS['M+K'] = lambda mz: (mz + 38.963158)
POS_TRANSFORMATIONS['M+2Na-H'] = lambda mz: (mz + 44.971160)
POS_TRANSFORMATIONS['M+ACN+Na'] = lambda mz: (mz + 64.015765)
POS_TRANSFORMATIONS['M+2Na-H'] = lambda mz: (mz + 44.971160)
POS_TRANSFORMATIONS['M+2K+H'] = lambda mz: (mz + 76.919040)
POS_TRANSFORMATIONS['[M+DMSO]+H'] = lambda mz: (mz + 79.02122)
POS_TRANSFORMATIONS['[M+2ACN]+H'] = lambda mz: (mz + 83.060370)
POS_TRANSFORMATIONS['2M+H'] = lambda mz: (mz * 2) + 1.007276
POS_TRANSFORMATIONS['M+ACN+Na'] = lambda mz: (mz + 64.015765)
POS_TRANSFORMATIONS['2M+NH4'] = lambda mz: (mz * 2) + 18.

NEG_TRANSFORMATIONS = collections.OrderedDict()
NEG_TRANSFORMATIONS['M-H'] = lambda mz: (mz - PROTON_MASS)

# example prior dictionary to be passed when creating an
# adducts object to only get M+H adducts out
ADDUCT_DICT_POS_MH = {POSITIVE: {'M+H': 1.0}}

ATOM_NAMES = ['C', 'H', 'N', 'O', 'P', 'S', 'Cl', 'I', 'Br', 'Si', 'F', 'D']
ATOM_MASSES = {'C': 12.00000000000, 'H': 1.00782503214, 'O': 15.99491462210,
               'N': 14.00307400524,
               'P': 30.97376151200, 'S': 31.97207069000, 'Cl': 34.96885271000,
               'I': 126.904468, 'Br': 78.9183376,
               'Si': 27.9769265327, 'F': 18.99840320500, 'D': 2.01410177800}

GET_MS2_BY_PEAKS = "sample"
GET_MS2_BY_SPECTRA = "spectra"
DUMMY_PRECURSOR_MZ = 123.456

INITIAL_SCAN_ID = 100000
DEFAULT_SCAN_TIME_DICT = {1: 0.4, 2: 0.2}
DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS = {
    'min_intensity': 1.75E5,
    'mz_tol': 10,
    'mz_units': 'ppm',
    'min_length': 2,
    'start_rt': 0,
    'stop_rt': 1440,
}

MSDIAL_DDA_MODE = 'lcmsdda'
MSDIAL_DIA_MODE = 'lcmsdia'

IN_SILICO_OPTIMISE_TOPN = 'TopN'
IN_SILICO_OPTIMISE_SMART_ROI = 'SmartROI'
IN_SILICO_OPTIMISE_WEIGHTED_DEW = 'WeightedDEW'

ROI_EXCLUSION_DEW = 'exclusion_dew'
ROI_EXCLUSION_WEIGHTED_DEW = 'exclusion_weighted_dew'

GRID_CONTROLLER_SCORING_PARAMS = {
    'theta1': 1,
    'theta2': 0,
    'theta3': 0,
    'theta4': 0
}

ROI_TYPE_NORMAL = 'roi'
ROI_TYPE_SMART = 'smart'


###############################################################################
# Common classes
###############################################################################


class Formula():
    """
    A class to represent a chemical formula
    """

    def __init__(self, formula_string):
        """
        Creates a Formula object. Formulae can be sampled to generate [vimms.Chemicals.Chemical][]
        objects.

        Args:
            formula_string (string): the chemical formula
        """
        self.formula_string = formula_string
        self.atom_names = ATOM_NAMES
        self.atoms = {}
        for atom in self.atom_names:
            self.atoms[atom] = self._get_n_element(atom)
        self.mass = self._get_mz()

    def _get_mz(self):
        """
        Computes the m/z value of this formula
        Returns: the m/z value

        """
        # Assume no charge
        return self.compute_exact_mass()

    def _get_n_element(self, atom_name):
        """
        Computes how many elements of an atom is present in the formula
        Args:
            atom_name: the atom to check

        Returns: the total number of atoms present in the formula

        """
        # Do some regex matching to find the numbers of the important atoms
        ex = atom_name + '(?![a-z])' + '\\d*'
        m = re.search(ex, self.formula_string)
        if m is None:
            return 0
        else:
            ex = atom_name + '(?![a-z])' + '(\\d*)'
            m2 = re.findall(ex, self.formula_string)
            total = 0
            for a in m2:
                if len(a) == 0:
                    total += 1
                else:
                    total += int(a)
            return total

    def compute_exact_mass(self):
        """
        Computes the exact mass of this formula
        Returns: the exact mass

        """
        masses = ATOM_MASSES
        exact_mass = 0.0
        for a in self.atoms:
            exact_mass += masses[a] * self.atoms[a]
        return exact_mass

    def __repr__(self):
        return self.formula_string

    def __str__(self):
        return self.formula_string


class DummyFormula():
    """
    A dummy wrapper to store an mz as a [vimms.Common.Formula][].
    This is convenient as it allows us to treat an m/z value like an (unknown) formula.
    """

    def __init__(self, mz):
        """
        Create a DummyFormula object
        Args:
            mz: the m/z value to wrap
        """
        self.mass = mz

    def compute_exact_mass(self):
        return self.mass


class ScanParameters():
    """
    A class to store parameters used to instruct the mass spec how to
    generate a scan. This object is usually created by the controller.
    It is used by the controller to instruct the mass spec what actions (scans)
    to perform next.
    """

    MS_LEVEL = 'ms_level'
    COLLISION_ENERGY = 'collision_energy'
    POLARITY = 'polarity'
    FIRST_MASS = 'first_mass'
    LAST_MASS = 'last_mass'
    ORBITRAP_RESOLUTION = 'orbitrap_resolution'
    AGC_TARGET = 'agc_target'
    MAX_IT = 'max_it'
    MASS_ANALYSER = 'analyzer'
    ACTIVATION_TYPE = 'activation_type'
    ISOLATION_MODE = 'isolation_mode'
    SOURCE_CID_ENERGY = 'source_cid_energy'
    METADATA = 'metadata'
    UNIQUENESS_TOKEN = "uniqueness_token"

    # this is used for DIA-based controllers to specify which windows to
    # fragment
    ISOLATION_WINDOWS = 'isolation_windows'

    # precursor m/z and isolation width have to be specified together for
    # DDA-based controllers
    PRECURSOR_MZ = 'precursor_mz'
    ISOLATION_WIDTH = 'isolation_width'

    # used in Top-N, hybrid and ROI controllers to perform dynamic exclusion
    DYNAMIC_EXCLUSION_MZ_TOL = 'dew_mz_tol'
    DYNAMIC_EXCLUSION_RT_TOL = 'dew_rt_tol'

    # only used by the hybrid controller for now, since its Top-N may change
    # depending on time for other DDA controllers it's always the same
    # throughout the whole run, so we don't send this parameter
    CURRENT_TOP_N = 'current_top_N'

    # if the scan id is specified, then it should be used by the mass spec
    # useful for pre-scheduled controllers where we want the controller
    # to know the scan ids of MS1, MS2
    # and also the precursor ids of those MS2 scans in advance.
    SCAN_ID = 'scan_id'

    def __init__(self):
        """
        Create a scan parameter object
        """
        self.params = {}

    def set(self, key, value):
        """
        Set scan parameter value

        Args:
            key: a scan parameter name
            value: a scan parameter value

        Returns: None
        """
        self.params[key] = value

    def get(self, key):
        """
        Gets scan parameter value

        Args:
            key: the key to look for

        Returns: the corresponding value in this ScanParameter
        """
        if key in self.params:
            return self.params[key]
        else:
            return None

    def get_all(self):
        """
        Get all scan parameters
        Returns: all the scan parameters
        """
        return self.params

    def compute_isolation_windows(self):
        """
        Gets the full-width (DDA) isolation window around a precursor m/z
        """
        precursor_list = self.get(ScanParameters.PRECURSOR_MZ)
        isolation_width_list = self.get(ScanParameters.ISOLATION_WIDTH)

        isolation_windows = []

        windows = []
        for i, precursor in enumerate(precursor_list):
            isolation_width = isolation_width_list[i]
            mz_lower = precursor.precursor_mz - (isolation_width / 2)
            mz_upper = precursor.precursor_mz + (isolation_width / 2)
            windows.append((mz_lower, mz_upper))

        isolation_windows.append(windows)
        return isolation_windows

    def __repr__(self):
        return 'ScanParameters %s' % (self.params)


class Precursor():
    """
    A class to store precursor peak information when writing an MS2 scan.
    """

    def __init__(self, precursor_mz, precursor_intensity, precursor_charge,
                 precursor_scan_id):
        """
        Create a Precursor object.
        Args:
            precursor_mz: the m/z value of this precursor peak.
            precursor_intensity: the intensity value of this precursor peak.
            precursor_charge: the charge of this precursor peak
            precursor_scan_id: the assocated MS1 scan ID that contains this precursor peak
        """
        self.precursor_mz = precursor_mz
        self.precursor_intensity = precursor_intensity
        self.precursor_charge = precursor_charge
        self.precursor_scan_id = precursor_scan_id

    def __repr__(self):
        return 'Precursor mz %f intensity %f charge %d scan_id %d' % (
            self.precursor_mz, self.precursor_intensity, self.precursor_charge,
            self.precursor_scan_id)


###############################################################################
# Common methods
###############################################################################

def create_if_not_exist(out_dir):
    """
    Creates a directory if it doesn't already exist
    Args:
        out_dir: the directory to create, if it doesn't exist

    Returns: None.

    """
    if not pathlib.Path(out_dir).exists():
        logger.info('Created %s' % out_dir)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        
def path_or_mzml(mzml):    
    try:
        mzml = MZMLFile(mzml)
    except:
        if(not type(mzml) == MZMLFile):
            raise NotImplementedError("Didn't recognise the MZMLFile!")
    return mzml

def save_obj(obj, filename):
    """
    Save object to file. This is useful for storing simulation results and other objects.

    If the directory containing the specified filename doesn't exist, it will be created first.
    The object will be saved using gzip + pickle (highest protocol).

    Args:
        obj: the Python object to save
        filename: the output filename to use

    Returns: None
    """

    # workaround for
    # TypeError: can't pickle _thread.lock objects
    # when trying to pickle a progress bar
    if hasattr(obj, 'bar'):
        obj.bar = None

    out_dir = os.path.dirname(filename)
    create_if_not_exist(out_dir)
    logger.info('Saving %s to %s' % (type(obj), filename))
    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """
    Load saved object from file

    Args:
        filename: The filename to load. Should be saved using the `save_obj` method.

    Returns: the loaded object

    """
    try:
        with gzip.GzipFile(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. '
                       'Please regenerate this file.' % filename)
        raise


def chromatogramDensityNormalisation(rts, intensities):
    """
    Definition to standardise the area under a chromatogram to 1.

    Args:
        rts: the RT values of this chromatogram
        intensities: the intensity values of this chromatogram

    Returns: updated intensities that have been standardised so the area is 1.

    """
    assert len(rts) == len(intensities)
    area = 0.0
    for rt_index in range(len(rts) - 1):
        area += ((intensities[rt_index] + intensities[rt_index + 1]) / 2) / (
                rts[rt_index + 1] - rts[rt_index])
    new_intensities = [x * (1 / area) for x in intensities]
    return new_intensities


def adduct_transformation(mz, adduct):
    """
    Transform m/z value according to the selected adduct transformation.

    Args:
        mz: the m/z value to check
        adduct: the selected adduct transformation

    Returns: the new m/z value for the adduct

    """
    if adduct in POS_TRANSFORMATIONS:
        f = POS_TRANSFORMATIONS[adduct]
    elif adduct in NEG_TRANSFORMATIONS:
        f = NEG_TRANSFORMATIONS[adduct]
    else:
        def f(mz):
            return mz
    return f(mz)


def take_closest(my_list, my_number):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    Args:
        my_list: the list to check
        my_number: the number to find in the list

    Returns: The closest value to myNumber. If two numbers are equally close,
    return the smallest number.
    """
    pos = bisect_left(my_list, my_number)
    if pos == 0:
        return 0
    if pos == len(my_list):
        return -1
    before = my_list[pos - 1]
    after = my_list[pos]
    if after - my_number < my_number - before:
        return pos
    else:
        return pos - 1


def set_log_level(level, remove_id=None):
    """
    Set the logging level of the default logger
    Args:
        level: the new level to set
        remove_id: ID of previous log handler to be removed

    Returns: the new log handler after setting the log level

    """
    if remove_id is None:
        try:
            logger.remove(0)  # try to remove the default handler with id 0
        except ValueError:  # no default handler has been set
            pass
    else:
        logger.remove(remove_id)  # remove previously set handler by id

    # add new handler at the desired log level
    new_handler_id = logger.add(sys.stderr, level=level)
    return new_handler_id


def set_log_level_warning(remove_id=None):
    """
    Set log level to WARNING
    Args:
        remove_id: ID of previous log handler to be removed

    Returns: None

    """
    return set_log_level(logging.WARNING, remove_id=remove_id)


def set_log_level_info(remove_id=None):
    """
    Set log level to INFO
    Args:
        remove_id: ID of previous log handler to be removed

    Returns: None

    """
    return set_log_level(logging.INFO, remove_id=remove_id)


def set_log_level_debug(remove_id=None):
    """
    Set log level to DEBUG
    Args:
        remove_id: ID of previous log handler to be removed

    Returns: None

    """
    return set_log_level(logging.DEBUG, remove_id=remove_id)


def add_log_file(log_path, level):
    """
    Add path to log file
    Args:
        log_path: filename to output the logging to
        level: the log level

    Returns: None

    """
    logger.add(log_path, level=level)


def get_rt(spectrum):
    """
    Extracts RT value from a pymzml spectrum object

    Args:
        spectrum: a pymzml spectrum object

    Returns: the retention time (in seconds)

    """
    rt, units = spectrum.scan_time
    if units == 'minute':
        rt *= 60.0
    return rt


def find_nearest_index_in_array(array, value):
    """
    Finds index in array where the value is the nearest

    Args:
        array: the array to check
        value: the value to check

    Returns: index in array where the value is the nearest

    """
    idx = (np.abs(array - value)).argmin()
    return idx


def download_file(url, out_file=None):
    """
    Download a file from the given URL
    Args:
        url: URL to download
        out_file: filename of output file to save to

    Returns: filename of the out_file

    """
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    current_size = 0

    if out_file is None:
        out_file = url.rsplit('/', 1)[-1]  # get the last part in url
    logger.info('Downloading %s' % out_file)

    with open(out_file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size // block_size), unit='KB',
                         unit_scale=True):
            current_size += len(data)
            f.write(data)
    assert current_size == total_size
    return out_file


def extract_zip_file(in_file, delete=True):
    """
    Extract a zip file
    Args:
        in_file: the input zip file
        delete: whether to delete the input zip file after extracting

    Returns: None

    """
    logger.info('Extracting %s' % in_file)
    with zipfile.ZipFile(file=in_file) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(),
                         total=len(zip_file.namelist())):
            zip_file.extract(member=file)

    if delete:
        logger.info('Deleting %s' % in_file)
        os.remove(in_file)


def uniform_list(N, min_val, max_val):
    """
    Generates a list of N uniformly random values from min_val to max_val
    Args:
        N: the number of items to generate
        min_val: the minimum range
        max_val: the maximum range

    Returns: a list of N values

    """
    return list(np.random.rand(N) * (max_val - min_val) + min_val)


def get_default_scan_params(
        polarity=POSITIVE,
        agc_target=DEFAULT_MS1_AGC_TARGET,
        max_it=DEFAULT_MS1_MAXIT,
        collision_energy=DEFAULT_MS1_COLLISION_ENERGY,
        source_cid_energy=DEFAULT_SOURCE_CID_ENERGY,
        orbitrap_resolution=DEFAULT_MS1_ORBITRAP_RESOLUTION,
        default_ms1_scan_window=DEFAULT_MS1_SCAN_WINDOW,
        mass_analyser=DEFAULT_MS1_MASS_ANALYSER,
        activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
        isolation_mode=DEFAULT_MS1_ISOLATION_MODE,
        metadata=None,
        scan_id=None):
    """
    Generate the default MS1 scan parameters.

    Args:
        polarity: the polarity value, either POSITIVE or NEGATIVE
        agc_target: AGC (automatic gain control) target
        max_it: maximum time to collect ion
        collision_energy: the collision energy to use
        source_cid_energy: source CID energy
        orbitrap_resolution: resolution of the mass-spec (Orbitrap) instrument
        default_ms1_scan_window: the default MS1 scan window
        mass_analyser: which mass analyser to use
        activation_type: activation type, either HCD or CID
        isolation_mode: isolation mode, either None, or Quadrupole or IonTrap
        metadata: additional metadata to include in this scan
        scan_id: the scan ID, if specified

    Returns: the parameters of the MS1 scan to create

    """
    default_scan_params = ScanParameters()
    default_scan_params.set(ScanParameters.MS_LEVEL, 1)
    default_scan_params.set(ScanParameters.ISOLATION_WINDOWS,
                            [[default_ms1_scan_window]])
    default_scan_params.set(ScanParameters.ISOLATION_WIDTH,
                            DEFAULT_ISOLATION_WIDTH)
    default_scan_params.set(ScanParameters.COLLISION_ENERGY, collision_energy)
    default_scan_params.set(ScanParameters.ORBITRAP_RESOLUTION,
                            orbitrap_resolution)
    default_scan_params.set(ScanParameters.ACTIVATION_TYPE, activation_type)
    default_scan_params.set(ScanParameters.MASS_ANALYSER, mass_analyser)
    default_scan_params.set(ScanParameters.ISOLATION_MODE, isolation_mode)
    default_scan_params.set(ScanParameters.AGC_TARGET, agc_target)
    default_scan_params.set(ScanParameters.MAX_IT, max_it)
    default_scan_params.set(
        ScanParameters.SOURCE_CID_ENERGY, source_cid_energy)
    default_scan_params.set(ScanParameters.POLARITY, polarity)
    default_scan_params.set(ScanParameters.FIRST_MASS,
                            default_ms1_scan_window[0])
    default_scan_params.set(ScanParameters.LAST_MASS,
                            default_ms1_scan_window[1])
    default_scan_params.set(ScanParameters.METADATA, metadata)
    default_scan_params.set(ScanParameters.SCAN_ID, scan_id)
    return default_scan_params


def get_dda_scan_param(mz, intensity, precursor_scan_id, isolation_width,
                       mz_tol, rt_tol,
                       agc_target=DEFAULT_MS2_AGC_TARGET,
                       max_it=DEFAULT_MS2_MAXIT,
                       collision_energy=DEFAULT_MS2_COLLISION_ENERGY,
                       source_cid_energy=DEFAULT_SOURCE_CID_ENERGY,
                       orbitrap_resolution=DEFAULT_MS2_ORBITRAP_RESOLUTION,
                       mass_analyser=DEFAULT_MS2_MASS_ANALYSER,
                       activation_type=DEFAULT_MS1_ACTIVATION_TYPE,
                       isolation_mode=DEFAULT_MS2_ISOLATION_MODE,
                       polarity=POSITIVE,
                       metadata=None, scan_id=None):
    """
    Generate the default MS2 scan parameters.

    Args:
        mz: m/z of precursor peak to fragment
        intensity: intensity of precursor peak to fragment
        precursor_scan_id: scan ID of the MS1 scan containing the precursor peak
        isolation_width: isolation width, in Dalton
        mz_tol: m/z tolerance for dynamic exclusion # FIXME: this shouldn't be here
        rt_tol: RT tolerance for dynamic exclusion # FIXME: this shouldn't be here
        agc_target: AGC (automatic gain control) target
        max_it: maximum time to collect ion
        collision_energy: the collision energy to use
        source_cid_energy: source CID energy
        orbitrap_resolution: resolution of the mass-spec (Orbitrap) instrument
        mass_analyser: which mass analyser to use
        activation_type: activation type, either HCD or CID
        isolation_mode: isolation mode, either None, or Quadrupole or IonTrap
        polarity: the polarity value, either POSITIVE or NEGATIVE
        metadata: additional metadata to include in this scan
        scan_id: the scan ID, if specified

    Returns: the parameters of the MS2 scan to create

    """

    dda_scan_params = ScanParameters()
    dda_scan_params.set(ScanParameters.MS_LEVEL, 2)

    assert isinstance(mz, list) == isinstance(intensity, list)

    # create precursor object, assume it's all singly charged
    precursor_charge = +1 if (polarity == POSITIVE) else -1
    if type(mz) == list:
        precursor_list = []
        for i, m in enumerate(mz):
            precursor_list.append(
                Precursor(precursor_mz=m, precursor_intensity=intensity[i],
                          precursor_charge=precursor_charge,
                          precursor_scan_id=precursor_scan_id))
        dda_scan_params.set(ScanParameters.PRECURSOR_MZ, precursor_list)

        if type(isolation_width) == list:
            assert len(isolation_width) == len(precursor_list)
        else:
            isolation_width = [isolation_width for m in mz]
        dda_scan_params.set(ScanParameters.ISOLATION_WIDTH, isolation_width)

    else:
        precursor = Precursor(precursor_mz=mz, precursor_intensity=intensity,
                              precursor_charge=precursor_charge,
                              precursor_scan_id=precursor_scan_id)
        precursor_list = [precursor]
        dda_scan_params.set(ScanParameters.PRECURSOR_MZ, precursor_list)

        # set the full-width isolation width, in Da
        # if mz is not a list, neither should isolation_width be
        assert not isinstance(isolation_width, list)
        isolation_width = [isolation_width]
        dda_scan_params.set(ScanParameters.ISOLATION_WIDTH, isolation_width)

    # define dynamic exclusion parameters
    dda_scan_params.set(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL, mz_tol)
    dda_scan_params.set(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL, rt_tol)

    # define other fragmentation parameters
    dda_scan_params.set(ScanParameters.COLLISION_ENERGY, collision_energy)
    dda_scan_params.set(ScanParameters.ORBITRAP_RESOLUTION,
                        orbitrap_resolution)
    dda_scan_params.set(ScanParameters.ACTIVATION_TYPE, activation_type)
    dda_scan_params.set(ScanParameters.MASS_ANALYSER, mass_analyser)
    dda_scan_params.set(ScanParameters.ISOLATION_MODE, isolation_mode)
    dda_scan_params.set(ScanParameters.AGC_TARGET, agc_target)
    dda_scan_params.set(ScanParameters.MAX_IT, max_it)
    dda_scan_params.set(ScanParameters.SOURCE_CID_ENERGY, source_cid_energy)
    dda_scan_params.set(ScanParameters.POLARITY, polarity)
    dda_scan_params.set(ScanParameters.FIRST_MASS, DEFAULT_MSN_SCAN_WINDOW[0])
    dda_scan_params.set(ScanParameters.METADATA, metadata)
    dda_scan_params.set(ScanParameters.SCAN_ID, scan_id)

    # dynamically scale the upper mass
    charge = 1
    wiggle_room = 1.1
    max_precursor_mz = max([(p.precursor_mz + isol / 2) for (p, isol) in
                            zip(precursor_list, isolation_width)])
    last_mass = max_precursor_mz * charge * wiggle_room
    dda_scan_params.set(ScanParameters.LAST_MASS, last_mass)
    return dda_scan_params
