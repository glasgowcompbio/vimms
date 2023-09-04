---
nav_exclude: true
---
# Creating Chemicals

Chemicals are the essential components run through the simulated LC-MS/MS process in ViMMS. There are two main types: [`Known Chemicals`][vimms.Chemicals.KnownChemical] and [`Unknown Chemicals`][vimms.Chemicals.UnknownChemical].

## Known Chemicals

Known chemicals refer to substances with identified properties and are represented by their formulae. They can be sampled from databases such as HMDB or from specific distributions using classes extending the [`Formula Sampler`][vimms.ChemicalSamplers.FormulaSampler]. Current options include:

- [`DatabaseFormulaSampler`][vimms.ChemicalSamplers.DatabaseFormulaSampler]: samples formulas from a provided database.
- [`UniformMZFormulaSampler`][vimms.ChemicalSamplers.UniformMZFormulaSampler]: samples formulas uniformly in a defined m/z range.
- [`PickEverythingFormulaSampler`][vimms.ChemicalSamplers.PickEverythingFormulaSampler]: samples all formulas from a database.
- [`EvenMZFormulaSampler`][vimms.ChemicalSamplers.EvenMZFormulaSampler]: creates evenly spaced m/z, primarily for test cases.
- [`MZMLFormulaSampler`][vimms.ChemicalSamplers.MZMLFormulaSampler]: samples m/z values from a histogram of m/z derived from a user-supplied mzML file.

After generating a list of formula objects, you can use the [`Chemical Mixture Creator`][vimms.Chemicals.ChemicalMixtureCreator] class to produce chemical datasets for simulation:

```python
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.ChemicalSamplers import UniformMZFormulaSampler

df = UniformMZFormulaSampler(min_mz=100, max_mz=500)
cm = ChemicalMixtureCreator(df)
chemicals = cm.sample(100, 2)  # sample 100 chemicals up to MS2
```

ViMMS offers many options to specify formula samplers and customize the generated RT, intensity, chromatograms, and MS2 peaks for a Chemical object. 

You can explore these functionalities further with our notebooks demonstrating the creation of [purely simulated chemicals](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/03.%20Generating%20Sets%20of%20Chemicals%20with%20the%20ChemicalMixtureCreator%20class.ipynb) and [HMDB-sampled chemicals](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/01.%20Extracting%20Chemicals%20from%20HMDB.ipynb).

## Unknown Chemicals

Unknown chemicals are those without identifiable properties, typically extracted from existing mzML files. These could come from prior runs on an actual mass spectrometer. Each peak picked is presumed to correspond to a chemical, and their identities remain unknown. As fragmentation strategies operate without needing to know chemical identities, this presumption suffices for our simulation process.

For an example of how to extract unknown chemicals from existing mzML files, see this [notebook](https://github.com/glasgowcompbio/vimms/blob/master/demo/01.%20Data/02.%20Extracting%20Chemicals%20from%20an%20mzML%20file.ipynb).