import os
import random as rand
import time
from pathlib import Path

import numpy as np
from loguru import logger

from vimms.Chemicals import ChemSet
from vimms.Common import POSITIVE, load_obj
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer

MZ_RANGE = [(0, 1050)]
RT_RANGE = [(0, 1200)]
N_CHEMS = 10

np.random.seed(0)
rand.seed(0)


def generate_chems():
    DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    BASE_DIR = os.path.abspath(Path(DIR_PATH, '..', 'tests', 'fixtures'))
    BEER_CHEMS_PATH = Path(BASE_DIR, 'beer_compounds.p')
    BEER_CHEMS = load_obj(BEER_CHEMS_PATH)

    chemset = ChemSet.to_chemset(BEER_CHEMS, fast=True)
    return chemset


def run_topN(chemset):
    isolation_width = 1
    N = 10
    rt_tol = 15
    mz_tol = 10
    ionisation_mode = POSITIVE
    MIN_MS1_INTENSITY = 1

    # create a simulated mass spec without noise and Top-N controller
    mass_spec = IndependentMassSpectrometer(ionisation_mode, chemset)
    controller = TopNController(ionisation_mode, N, isolation_width, mz_tol, rt_tol,
                                MIN_MS1_INTENSITY)

    # create an environment to run both the mass spec and controller
    env = Environment(mass_spec, controller, 0, 1440, progress_bar=False)
    start_time = time.time()
    env.run()
    logger.info('Done in %s seconds' % (time.time() - start_time))


def main():
    large_fragscan_dataset = generate_chems()
    chemset = ChemSet.to_chemset(large_fragscan_dataset, fast=True)
    run_topN(chemset)


if __name__ == "__main__":
    main()
