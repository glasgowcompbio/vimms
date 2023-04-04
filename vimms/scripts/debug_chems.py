import os
import random as rand

import numpy as np

from vimms.ChemicalSamplers import DatabaseFormulaSampler
from vimms.ChemicalSamplers import GaussianChromatogramSampler
from vimms.ChemicalSamplers import UniformRTAndIntensitySampler, UniformMZFormulaSampler
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.Common import load_obj, POSITIVE, save_obj, set_log_level_warning
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.MassSpec import IndependentMassSpectrometer

np.random.seed(0)
rand.seed(0)

MZ_RANGE = [(0, 1050)]
RT_RANGE = [(0, 1440)]
N_CHEMS = 2

out_dir = 'debug_chems'

def generate_synthetic_chems(file_path):
    if os.path.exists(file_path):
        dataset = load_obj(file_path)
    else:
        min_mz = MZ_RANGE[0][0]
        max_mz = MZ_RANGE[0][1]
        min_rt = RT_RANGE[0][0]+500
        max_rt = RT_RANGE[0][1]-500
        um = UniformMZFormulaSampler(min_mz=min_mz, max_mz=max_mz)
        ri = UniformRTAndIntensitySampler(min_rt=min_rt, max_rt=max_rt,
                                          min_log_intensity=np.log(1e5))
        cs = GaussianChromatogramSampler(sigma=100)
        cm = ChemicalMixtureCreator(um, rt_and_intensity_sampler=ri, chromatogram_sampler=cs)
        dataset = cm.sample(N_CHEMS, 2)
        save_obj(dataset, file_path)
    return dataset


def generate_hmdb_chems(file_path):
    if os.path.exists(file_path):
        dataset = load_obj(file_path)
    else:
        compound_file = '/Users/joewandy/Work/git/vimms/tests/fixtures/hmdb_compounds.p'
        hmdb_compounds = load_obj(compound_file)
        df = DatabaseFormulaSampler(hmdb_compounds, min_mz=100, max_mz=1000)
        cm = ChemicalMixtureCreator(df)
        dataset = cm.sample(N_CHEMS, 2)
        save_obj(dataset, file_path)
    return dataset


min_rt = RT_RANGE[0][0]
max_rt = RT_RANGE[0][1]
isolation_window = 1
N = 3
rt_tol = 15
mz_tol = 10
min_ms1_intensity = 5000

set_log_level_warning()

# run for purely synthetic chems
file_path = os.path.join(out_dir, 'synthetic_chems.p')
dataset = generate_synthetic_chems(file_path)
print(dataset)
mass_spec = IndependentMassSpectrometer(POSITIVE, dataset)
controller = TopNController(POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)
env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True, out_dir=out_dir,
                  out_file="synthetic_chems.mzML")
env.run()
print(len(controller.scans[1]), len(controller.scans[2]))

# run for hmdb-sampled chems
file_path = os.path.join(out_dir, 'hmdb_chems.p')
dataset = generate_hmdb_chems(file_path)
mass_spec = IndependentMassSpectrometer(POSITIVE, dataset)
controller = TopNController(POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)
env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True, out_dir=out_dir,
                  out_file="hmdb_chems.mzML")
env.run()
print(len(controller.scans[1]), len(controller.scans[2]))
