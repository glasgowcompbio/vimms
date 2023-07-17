---
nav_exclude: true
---
# Creating Chemicals

Chemical objects are the input run through simulated LC-MS/MS process in ViMMS simulation. There are two types of
chemicals: [`Known Chemicals`][vimms.Chemicals.KnownChemical] and [`Unknown Chemicals`][vimms.Chemicals.UnknownChemical]

## Known Chemicals

Known chemicals refer to chemicals for which we know their identities. Known chemicals are represented by their
formulae. These can be sampled from databases of known chemicals (such as HMDB) or from certain distributions using
instance of a class extending [`Formula Sampler`][vimms.ChemicalSamplers.FormulaSampler]. Currently, the following
options are available:

- [`DatabaseFormulaSampler`][vimms.ChemicalSamplers.DatabaseFormulaSampler]: samples formulas from a database provided.
  Code is given to generate from HMDB

- [`UniformMZFormulaSampler`][vimms.ChemicalSamplers.UniformMZFormulaSampler]: samples formulas (just masses) uniformly
  in an m/z range

- [`PickEverythingFormulaSampler`][vimms.ChemicalSamplers.PickEverythingFormulaSampler]: samples all formulas from a
  database

- [`EvenMZFormulaSampler`][vimms.ChemicalSamplers.EvenMZFormulaSampler]: creates evenly spaced m/z, starting at 100Da,
  with 100 Da spacing. Mainly used for test cases.

- [`MZMLFormulaSampler`][vimms.ChemicalSamplers.MZMLFormulaSampler]: samples m/z values from a histogram of m/z taken
  from a user supplied mzML file

Once the list of formula objects have been created,
the [`Chemical Mixture Creator`][vimms.Chemicals.ChemicalMixtureCreator] class offers the most simple method for
creating chemical datasets for simulation allowing the generation of a dataset of chemicals in just a few lines of
Python. In the simplest case, we generate a list of formulae and pass them
to [`Chemical Mixture Creator`][vimms.Chemicals.ChemicalMixtureCreator] using all the default parameters.

```python
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.ChemicalSamplers import UniformMZFormulaSampler

df = UniformMZFormulaSampler(min_mz=100, max_mz=500) # samples m/z values uniformly between 100 and 500
cm = ChemicalMixtureCreator(df)
chemicals = cm.sample(100, 2)  # sample 100 chemicals up to MS2
```

Various options are provided in ViMMS to specify alternative formula samplers, as well as different options to customise
the generated RT, intensity, chromatograms and MS2 peaks for a Chemical object. For example:

```python
import os
from vimms.Common import load_obj
from vimms.Roi import RoiBuilderParams
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.ChemicalSamplers import DatabaseFormulaSampler, CRPMS2Sampler,
    MZMLChromatogramSampler

# specify ViMMS directory containing some test data
vimms_dir = '<specify the basedir of vimms here>'
data_dir = os.path.abspath(os.path.join(vimms_dir, 'tests', 'fixtures'))

# load parsed HMDB compounds in ViMMS
HMDB = os.path.join(data_dir, 'hmdb_compounds.p')
hmdb = load_obj(HMDB)

# load an example mzML file
MZML = os.path.join(data_dir, 'small_mzml.mzML')
roi_params = RoiBuilderParams(min_intensity=1000)
cs = MZMLChromatogramSampler(MZML, roi_params=roi_params)

# samples some chemicals using HMDB as the formula database
# with chromatograms coming from the specified mzML file 
# and generated MS2 peaks following the CRP process
df = DatabaseFormulaSampler(hmdb, min_mz=100, max_mz=1000)
cm = ChemicalMixtureCreator(df, ms2_sampler=CRPMS2Sampler(n_draws=100, alpha=2),
                            chromatogram_sampler=MZMLChromatogramSampler(MZML))
chemicals = cm.sample(100, 2)
```

The following notebooks demonstrate with more examples how chemicals can be created in these two
cases: [purely simulated chemicals](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/03.%20Generating%20Sets%20of%20Chemicals%20with%20the%20ChemicalMixtureCreator%20class.ipynb)
and
[HMDB-sampled chemicals](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/01.%20Extracting%20Chemicals%20from%20HMDB.ipynb)
.

## Unknown Chemicals

Unknown chemicals refer to chemicals for which we do not know their identities. This applies to chemicals extracted from
existing mzML files, which could come from previous runs on an actual mass spectrometry. In this case, chemicals are
extracted from the mzML file using peak picking. Each picked peak is assumed to correspond to a chemical, and the
identities of the chemicals in the system is assumed to be unknown. Since current fragmentation strategies operate
without the knowledge of chemical identities, this assumption is sufficient for our simulation process. To see an
example notebook demonstrating how unknown chemicals can be extracted from existing mzML
files, [see here](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/02.%20Extracting%20Chemicals%20from%20an%20mzML%20file.ipynb)
.