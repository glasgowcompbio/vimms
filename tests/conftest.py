import os
import random as rand
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from vimms.ChemicalSamplers import UniformMZFormulaSampler, UniformRTAndIntensitySampler, \
    GaussianChromatogramSampler, EvenMZFormulaSampler, ConstantChromatogramSampler, \
    MZMLFormulaSampler, MZMLRTandIntensitySampler, MZMLChromatogramSampler
from vimms.Chemicals import ChemicalMixtureCreator, ChemicalMixtureFromMZML
from vimms.Common import load_obj, set_log_level_warning, set_log_level_debug, \
    ADDUCT_DICT_POS_MH, ScanParameters
from vimms.Roi import RoiBuilderParams

# define some useful constants

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.abspath(Path(DIR_PATH, 'fixtures'))
HMDB = load_obj(Path(BASE_DIR, 'hmdb_compounds.p'))
OUT_DIR = str(Path(DIR_PATH, 'results'))

MIN_MS1_INTENSITY = 1

MZ_RANGE = [(0, 1050)]
RT_RANGE = [(0, 1200)]
CENTRE_RANGE = 600
MIN_RT = RT_RANGE[0][0]
MAX_RT = RT_RANGE[0][1]
N_CHEMS = 10

BEER_CHEMS = load_obj(Path(BASE_DIR, 'QCB_22May19_1.p'))
BEER_MIN_BOUND = 550
BEER_MAX_BOUND = 650

MZML_FILE = Path(BASE_DIR, 'small_mzml.mzML')
MGF_FILE = Path(BASE_DIR, 'small_mgf.mgf')


# define some useful methods

def get_rt_bounds(dataset, centre):
    rts = [ds.rt for ds in dataset]
    min_bound = max([rt for rt in rts if rt < centre], default=centre) - 60
    max_bound = min([rt for rt in rts if rt > centre], default=centre) + 60
    return (min_bound, max_bound)


def run_environment(env):
    # set the log level to WARNING so we don't see too many messages when environment is running
    set_log_level_warning()
    # run the simulation
    logger.info('Running simulation')
    env.run()
    logger.info('Done')
    # set the log level back to DEBUG
    set_log_level_debug()


def check_mzML(env, out_dir, filename):
    out_file = os.path.join(out_dir, filename)
    logger.info('Writing out mzML')
    env.write_mzML(out_dir, filename)
    logger.info('Done')
    assert os.path.exists(out_file)


def check_non_empty_MS1(controller):
    return check_non_empty(controller, 1)


def check_non_empty_MS2(controller):
    return check_non_empty(controller, 2)


def check_non_empty(controller, ms_level):
    non_empty = 0
    for scan in controller.scans[ms_level]:
        if scan.num_peaks > 0:
            non_empty += 1
        if scan.ms_level == 2:
            assert scan.scan_params is not None
            assert scan.scan_params.get(ScanParameters.PRECURSOR_MZ) is not None
    assert non_empty > 0


# define the fixtures shared across all tests

@pytest.fixture(autouse=True)
def random_seed():
    """Reset numpy and random seed generator."""
    np.random.seed(0)
    rand.seed(0)


@pytest.fixture(scope="module")
def fullscan_dataset():
    np.random.seed(0)
    rand.seed(0)
    min_mz = MZ_RANGE[0][0]
    max_mz = MZ_RANGE[0][1]
    min_rt = RT_RANGE[0][0]
    max_rt = RT_RANGE[0][1]
    um = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
    ri = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt)
    cs = GaussianChromatogramSampler(sigma=100)
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(N_CHEMS, 1)


@pytest.fixture(scope="module")
def fragscan_dataset():
    np.random.seed(0)
    rand.seed(0)
    min_mz = MZ_RANGE[0][0]
    max_mz = MZ_RANGE[0][1]
    min_rt = RT_RANGE[0][0]
    max_rt = RT_RANGE[0][1]
    um = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
    ri = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt)
    cs = GaussianChromatogramSampler(sigma=100)
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(N_CHEMS, 2)


@pytest.fixture(scope="module")
def simple_dataset():
    np.random.seed(0)
    rand.seed(0)
    um = UniformMZFormulaSampler(min_mz=515, max_mz=516)
    ri = UniformRTAndIntensitySampler(min_rt=150, max_rt=160)
    cs = GaussianChromatogramSampler(sigma=100)
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(1, 2)


@pytest.fixture(scope="module")
def ten_chems():
    np.random.seed(0)
    rand.seed(0)
    um = UniformMZFormulaSampler(min_mz=MZ_RANGE[0][0], max_mz=MZ_RANGE[0][1])
    ri = UniformRTAndIntensitySampler(min_rt=200, max_rt=300)
    cs = GaussianChromatogramSampler()
    cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(10, 2)


@pytest.fixture(scope="module")
def two_fixed_chems():
    np.random.seed(0)
    rand.seed(0)
    em = EvenMZFormulaSampler()
    ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
    cs = ConstantChromatogramSampler()
    cm = ChemicalMixtureCreator(em, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(2, 2)


@pytest.fixture(scope="module")
def even_chems():
    np.random.seed(0)
    rand.seed(0)
    # four evenly spaced chems for more advanced SWATH testing
    em = EvenMZFormulaSampler()
    ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=101)
    cs = ConstantChromatogramSampler()
    cm = ChemicalMixtureCreator(em, rt_and_intensity_sampler=ri, chromatogram_sampler=cs,
                                adduct_prior_dict=ADDUCT_DICT_POS_MH)
    return cm.sample(4, 2)


@pytest.fixture(scope="module")
def chems_from_mzml():
    np.random.seed(0)
    rand.seed(0)
    roi_params = RoiBuilderParams(min_roi_intensity=10, min_roi_length=5)
    cm = ChemicalMixtureFromMZML(MZML_FILE, roi_params=roi_params)
    return cm.sample(None, 2)


@pytest.fixture(scope="module")
def chem_mz_rt_i_from_mzml():
    np.random.seed(0)
    rand.seed(0)
    fs = MZMLFormulaSampler(MZML_FILE)
    ri = MZMLRTandIntensitySampler(MZML_FILE)
    cs = MZMLChromatogramSampler(MZML_FILE)
    cm = ChemicalMixtureCreator(fs, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
    return cm.sample(500, 2)
