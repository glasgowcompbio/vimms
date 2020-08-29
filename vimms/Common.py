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
from tqdm import tqdm

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

# example prior dictionary to be passed when creating an 
# adducts object to only get M+H adducts out
ADDUCT_DICT_POS_MH = {'M+H': 1.0}

ATOM_NAMES = ['C', 'H', 'N', 'O', 'P', 'S', 'Cl', 'I', 'Br', 'Si', 'F', 'D']
ATOM_MASSES = {'C': 12.00000000000, 'H': 1.00782503214, 'O': 15.99491462210, 'N': 14.00307400524,
               'P': 30.97376151200, 'S': 31.97207069000, 'Cl': 34.96885271000, 'I': 126.904468, 'Br': 78.9183376,
               'Si': 27.9769265327, 'F': 18.99840320500, 'D': 2.01410177800}

GET_MS2_BY_PEAKS = "sample"
GET_MS2_BY_SPECTRA = "spectra"

INITIAL_SCAN_ID = 100000
DEFAULT_SCAN_TIME_DICT = {1: 0.4, 2: 0.2}
DEFAULT_MZML_CHEMICAL_CREATOR_PARAMS = {
    'min_intensity': 1.75E5,
    'mz_tol': 10,
    'mz_units': 'ppm',
    'min_length': 2,
    'start_rt': 0,
    'stop_rt': 1440,
    'n_peaks': 1
}


class Formula(object):
    def __init__(self, formula_string):
        self.formula_string = formula_string
        self.atom_names = ATOM_NAMES
        self.atoms = {}
        for atom in self.atom_names:
            self.atoms[atom] = self._get_n_element(atom)
        self.mass = self._get_mz()

    def _get_mz(self):
        return self.compute_exact_mass()

    def _get_n_element(self, atom_name):
        # Do some regex matching to find the numbers of the important atoms
        ex = atom_name + '(?![a-z])' + '\\d*'
        m = re.search(ex, self.formula_string)
        if m == None:
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
        masses = ATOM_MASSES
        exact_mass = 0.0
        for a in self.atoms:
            exact_mass += masses[a] * self.atoms[a]
        return exact_mass

    def __repr__(self):
        return self.formula_string

    def __str__(self):
        return self.formula_string


class DummyFormula(object):
    # wrapper to store an mz as a 'formula'
    def __init__(self, mz):
        self.mass = mz

    def compute_exact_mass(self):
        return self.mass


def create_if_not_exist(out_dir):
    if not os.path.exists(out_dir) and len(out_dir) > 0:
        logger.info('Created %s' % out_dir)
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


def save_obj(obj, filename):
    """
    Save object to file
    :param obj: the object to save
    :param filename: the output file
    :return: None
    """
    out_dir = os.path.dirname(filename)
    create_if_not_exist(out_dir)
    logger.info('Saving %s to %s' % (type(obj), filename))
    with gzip.GzipFile(filename, 'w') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """
    Load saved object from file
    :param filename: The file to load
    :return: the loaded object
    """
    try:
        with gzip.GzipFile(filename, 'rb') as f:
            return pickle.load(f)
    except OSError:
        logger.warning('Old, invalid or missing pickle in %s. Please regenerate this file.' % filename)
        return None


def chromatogramDensityNormalisation(rts, intensities):
    """
    Definition to standardise the area under a chromatogram to 1. Returns updated intensities
    """
    area = 0.0
    for rt_index in range(len(rts) - 1):
        area += ((intensities[rt_index] + intensities[rt_index + 1]) / 2) / (rts[rt_index + 1] - rts[rt_index])
    new_intensities = [x * (1 / area) for x in intensities]
    return new_intensities


def adduct_transformation(mz, adduct):
    f = POS_TRANSFORMATIONS[adduct]
    return f(mz)


def takeClosest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0
    if pos == len(myList):
        return -1
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return pos
    else:
        return pos - 1


def set_log_level_warning():
    logger.remove()
    logger.add(sys.stderr, level=logging.WARNING)


def set_log_level_info():
    logger.remove()
    logger.add(sys.stderr, level=logging.INFO)


def set_log_level_debug():
    logger.remove()
    logger.add(sys.stderr, level=logging.DEBUG)


def get_rt(spectrum):
    '''
    Extracts RT value from a pymzml spectrum object
    :param spectrum: a pymzml spectrum object
    :return: the retention time (in seconds)
    '''
    rt, units = spectrum.scan_time
    if units == 'minute':
        rt *= 60.0
    return rt


def find_nearest_index_in_array(array, value):
    '''
    Finds index in array where the value is the nearest
    :param array:
    :param value:
    :return:
    '''
    idx = (np.abs(array - value)).argmin()
    return idx


def download_file(url, out_file=None):
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    current_size = 0

    if out_file is None:
        out_file = url.rsplit('/', 1)[-1]  # get the last part in url
    logger.info('Downloading %s' % out_file)

    with open(out_file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size // block_size), unit='KB',
                         unit_scale=True):
            current_size += len(data)
            f.write(data)
    assert current_size == total_size
    return out_file


def extract_zip_file(in_file, delete=True):
    logger.info('Extracting %s' % in_file)
    with zipfile.ZipFile(file=in_file) as zip_file:
        for file in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist())):
            zip_file.extract(member=file)

    if delete:
        logger.info('Deleting %s' % in_file)
        os.remove(in_file)


def uniform_list(N, min_val, max_val):
    return list(np.random.rand(N) * (max_val - min_val) + min_val)
