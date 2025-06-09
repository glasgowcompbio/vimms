#!/usr/bin/env python
# coding: utf-8

# # 03. Generating Sets of Chemicals with the ChemicalMixtureCreator class

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from pathlib import Path
import numpy as np


# In[3]:


import os
import sys
sys.path.append('../..')


# In[4]:


import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from vimms.Chemicals import ChemicalMixtureCreator
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController
from vimms.Environment import Environment
from vimms.Common import POSITIVE, set_log_level_warning, set_log_level_debug


# ## Introduction
# 
# The `ChemicalMixtureCreator` offers the most simple method for creating chemical datasets for simulation allowing the generation of a dataset of chemicals in just a few lines of python.
# 
# The class requires several other classes to run:
# 
# - A class that can generate _formulas_
# - A class that can generate RT and intensity for formulas
# - A class that can generate chromatograms for formulas
# - A class that can generate MS2 spectra for formulas
# 
# ### `FormulaSampler`
# 
# An instance of a class extending `FormulaSampler` must be provided. This samples the chemicals that will appear. Note that `Formula` is abused here: in some cases the result is a chemical formula, but in others, just an m/z. Which it does has implications for the data generated (see below).  Currently, the following options are available:
# 
# - `DatabaseFormulaSampler`: samples formulas from a database provided. Code is given to generate from HMDB
# - `UniformMZFormulaSampler`: samples _formulas_ (just masses) uniformly in an m/z range
# - `PickEverythingFormulaSampler`: samples all formulas from a database
# - `EvenMZFormulaSampler`: creates evenly spaced m/z, starting at 100Da, with 100 Da spacing. Mainly used for test cases.
# - `MZMLFormulaSampler`: samples m/z values from a histogram of m/z taken from a user supplied mzML file
# 
# Examples of creating some of these objects

# #### `DatabaseFormulaSampler`

# In[6]:


from vimms.ChemicalSamplers import DatabaseFormulaSampler
from vimms.Common import load_obj


# In[7]:


data_dir = os.path.abspath(os.path.join(os.getcwd(),'..','..','tests','fixtures'))
HMDB = os.path.join(data_dir,'hmdb_compounds.p')
hmdb = load_obj(HMDB)


# In[8]:


compound_file = 'hmdb_compounds.p'
try:
    hmdb_compounds = load_obj(compound_file)
except FileNotFoundError:

    # download the entire HMDB metabolite database and extract chemicals from it
    # url = 'http://www.hmdb.ca/system/downloads/current/hmdb_metabolites.zip'
    # out_file = download_file(url)
    # compounds = extract_hmdb_metabolite(out_file, delete=True)
    # save_obj(compounds, compound_file)

    # above could be quite slow slow, so download a pre-processed result instead
    url = 'https://github.com/glasgowcompbio/vimms-data/raw/main/hmdb_compounds.p'
    download_file(url, compound_file)
    hmdb_compounds = load_obj(compound_file)


# In[9]:


# create a database formula sampler that will sample from HMDB with m/z between 100 and 1000
df = DatabaseFormulaSampler(hmdb_compounds, min_mz=100, max_mz=1000)
samples = df.sample(1000)
mz_list = [s[0].mass for s in samples]
plt.hist(mz_list)


# In[10]:


from vimms.ChemicalSamplers import UniformMZFormulaSampler
# create a formula sampler that samples masses uniformly between 100 and 500
df = UniformMZFormulaSampler(min_mz=100, max_mz=500)
samples = df.sample(1000)
mz_list = [s[0].mass for s in samples]
plt.hist(mz_list)


# In[11]:


from vimms.ChemicalSamplers import MZMLFormulaSampler
MZML = os.path.join(data_dir, 'small_mzml.mzML')
df = MZMLFormulaSampler(MZML)
samples = df.sample(1000)
mz_list = [s[0].mass for s in samples]
plt.hist(mz_list)


# ### `RTAndIntensitySampler`
# 
# Passing an instance of this is optional. If nothing is passed, it defaults to `UniformRTAndIntensitySampler`
# 
# Available:
# 
# - `UniformRTAndIntensitySampler`: samples RT and intensity independently from uniform distributions (note that intensity is unifrom in log space)
# - `MZMLRTandIntensitySampler`: samples RT and intensity independely from histograms produced from an mzML file
# 
# Examples:

# #### `UniformRTAndIntensitySampler`

# In[12]:


from vimms.ChemicalSamplers import UniformRTAndIntensitySampler
ri = UniformRTAndIntensitySampler(min_rt=100, max_rt=500, min_log_intensity=2, max_log_intensity=9)
rt_list = []
intensity_list = []
for i in range(1000):
    a,b = ri.sample(None) #argument is a formula, but is ignored at the moment
    rt_list.append(a)
    intensity_list.append(b)
plt.figure()
plt.hist(rt_list)
plt.figure()
plt.hist(intensity_list)


# #### `MZMLRTandIntensitySampler`

# In[13]:


from vimms.ChemicalSamplers import MZMLRTandIntensitySampler
ri = MZMLRTandIntensitySampler(MZML)

rt_list = []
intensity_list = []

for i in range(1000):
    a,b = ri.sample(None) #argument is a formula, but is ignored at the moment
    rt_list.append(a)
    intensity_list.append(b)

plt.figure()
plt.hist(rt_list)
plt.figure()
plt.hist(intensity_list)


# ### `ChromatogramSampler`
# 
# This optional object defines where chromatograms should be sampled from for each formula. There are three options:
# 
# - `GaussianChromatogramSampler`: generates normal shape chromatographic peaks
# - `ConstantChromoatogramSampler`: generates constant chromatographic (i.e. flat) peaks (mainly for testing)
# - `MZMLChromatogramSampelr`: samples chromatograms from ROIs extracted from an mzML file
# 
# Note that in all cases, the `sample` method takes three arguments: a formula, an rt and an intensity. These are so that, in future we could condition the chromatogram finding on particular values of RT and intensity (e.g. high intensity = better peaks).
# 
# Examples:

# #### `GaussianChomatogramSampler`

# In[14]:


# grab a formula to use for example sampling
f_list = df.sample(1)
formula, name = f_list[0]
from vimms.ChemicalSamplers import GaussianChromatogramSampler
cs = GaussianChromatogramSampler(sigma=10)
example_rt = 100
example_intensity = 1e5
c = cs.sample(formula, example_rt, example_intensity)
rt_vals = np.linspace(50,150)
intensities = []
for r in rt_vals:
    intensities.append(c.get_relative_intensity(r - example_rt))
plt.plot(rt_vals, intensities)


# #### `ConstantChromatogramSampler`

# In[15]:


from vimms.ChemicalSamplers import ConstantChromatogramSampler
cs = ConstantChromatogramSampler()
example_rt = 100
example_intensity = 1e5
c = cs.sample(formula, example_rt, example_intensity)
rt_vals = np.linspace(50,150)
intensities = []
for r in rt_vals:
    intensities.append(c.get_relative_intensity(r - example_rt))
plt.plot(rt_vals, intensities)


# #### `MZMLChromatogramSampler`

# In[63]:


# note that if you want to set the parameters for the ROI extraction from the mzML, use the RioParamsBuilder object
# e.g.
from vimms.Roi import RoiBuilderParams
from vimms.ChemicalSamplers import MZMLChromatogramSampler

roi_params = RoiBuilderParams(min_roi_intensity=10000)
cs = MZMLChromatogramSampler(MZML, roi_params=roi_params)
c = cs.sample(formula, example_rt, example_intensity)


# In[65]:


rt_vals = np.linspace(50, 150)
intensities = []
for r in rt_vals:
    intensities.append(c.get_relative_intensity(r - example_rt))
plt.plot(rt_vals, intensities)


# ### MS2Sampler
# 
# This final class determines how chemicals will be assigned MS2 peaks. There are six options:
# 
# - `UniformMS2Sampler`: samples uniformly between a min mass and the mass of the formula
# - `FixedMS2Sampler`: generates a fixed number of peaks, evenly spaced below the formula (mainly for testing)
# - `CRPSMS2Sampler`: generates MS2 peaks using a Chinese Restaurant Process
# - `MGFMS2Sampler`: generates MS2 spectra by sampling from those in an mgf file
# - `ExactMatchMS2Sampler`: to be used in the case where objects in the MGF file share an ID with the database used for formula sampling
# - `MZMLMS2Sampler`: samples MS2 spectra from an MS2 scan in an mzML file
# 
# In all cases, `sample` returns a list of masses, a list of intensties, and a *parent proportion* which is how much of the parents intensity gets transfered into the MS2 spectrum. The proportion is uniform between `min_proportion` and `max_proportion` which are passed to the constructors.
# 
# Examples:

# In[66]:


def plot_spectrum(mz_list, intensity_list):
    plt.figure()
    for i,m in enumerate(mz_list):
        plt.plot([m,m],[0,intensity_list[i]])
class TempChemical:
    def __init__(self, mass):
        self.mass = mass


# #### `UniformMS2Sampler`

# In[67]:


from vimms.ChemicalSamplers import UniformMS2Sampler
ms = UniformMS2Sampler(poiss_peak_mean=5) # number of fragments is decided by sample from poisson
tc = TempChemical(formula.mass)
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# #### `CRPMS2Sampler`

# In[68]:


from vimms.ChemicalSamplers import CRPMS2Sampler
ms = CRPMS2Sampler(n_draws=500, alpha=1) # alpha and n_draws control the propery of the CRP
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# #### `FixedMS2Sampler`

# In[69]:


from vimms.ChemicalSamplers import FixedMS2Sampler
ms = FixedMS2Sampler(n_frags=3) # how many to make
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# #### `MGFMS2Sampler`

# In[70]:


MGF = os.path.join(data_dir, 'small_mgf.mgf')
from vimms.ChemicalSamplers import MGFMS2Sampler
ms = MGFMS2Sampler(MGF)
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# #### `MZMLMS2Sampler`

# In[71]:


from vimms.ChemicalSamplers import MZMLMS2Sampler
ms = MZMLMS2Sampler(MZML)
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# #### `ExactMatchMS2Sampler`

# In[72]:


from vimms.ChemicalSamplers import ExactMatchMS2Sampler
# when formulas are sampled from a database, their accession is stored. We can cheat this as follows:
tc.database_accession = 'CCMSLIB00005435506'
# the MS2 sampler will then extract the spectrum that has the id_fiels set to this ID value
ms = ExactMatchMS2Sampler(MGF, id_field='SPECTRUMID')
a = ms.sample(tc)
mz_list = a[0]
intensity_list = a[1]
plot_spectrum(mz_list, intensity_list)


# ## Everything together - `ChemicalMixtureCreator`

# The simplest use is to just pass a formula sampler and let the rest go to defaults:

# In[73]:


from vimms.Chemicals import ChemicalMixtureCreator
df = DatabaseFormulaSampler(hmdb, min_mz=100, max_mz=1000)
cm = ChemicalMixtureCreator(df)
chemicals = cm.sample(100,2) # sample 100 chemicals up to MS2


# If more tailoring is required, pass the different samplers as arguments. E.g. if you wanted a CRPMS2Sampler and MZMLChromatograms:

# In[74]:


cm = ChemicalMixtureCreator(df, ms2_sampler=CRPMS2Sampler(n_draws=100, alpha=2), chromatogram_sampler=MZMLChromatogramSampler(MZML))
chemicals = cm.sample(100,2)


# ### Use in simulator

# We can use the sampled chemicals to simulate various fragmentation strategies in ViMMS. Below we run it through a TopN strategy.
# 
# First we set some parameters for the Top-N controller and its simulated environment.

# In[75]:


rt_range = [(0, 1440)]
min_rt = rt_range[0][0]
max_rt = rt_range[0][1]


# In[76]:


isolation_window = 1
N = 3
rt_tol = 15
mz_tol = 10
min_ms1_intensity = 1.75E5


# Initialise simulated mass spec and the Top-N controller 

# In[77]:


mass_spec = IndependentMassSpectrometer(POSITIVE, chemicals)
controller = TopNController(POSITIVE, N, isolation_window, mz_tol, rt_tol, min_ms1_intensity)


# Create an environment to run both the mass spec and controller. Set the log level to WARNING so we don't see too many messages when environment is running.

# In[78]:


set_log_level_warning()
env = Environment(mass_spec, controller, min_rt, max_rt, progress_bar=True)
env.run()


# Write the resulting mzML file from simulation to the location below. You can use ToppView from OpenMS or other mzML viewer to inspect the results. Note that the output wouldn't look very realistic as the chromatograms for all chemicals are the same (gaussian), and there's no noise or small peaks at all.

# In[79]:


set_log_level_debug()
mzml_filename = 'hmdb_topn_controller_2.mzML'
out_dir = os.path.join(os.getcwd(), 'results')
env.write_mzML(out_dir, mzml_filename)

