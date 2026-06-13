# Simulation Overview

This page summarises the typical workflow when using ViMMS to create new LC--MS/MS data. A simulation involves the following main components:

1. **Chemicals** – virtual representations of compounds to be ionised. See [Creating Chemicals](chemicals.md) for more details.
2. **Mass Spectrometer** – either an in silico model (`IndependentMassSpectrometer`) or a real instrument.
3. **Controller** – defines the fragmentation strategy, for example Top‑N DDA.
4. **Environment** – orchestrates interaction between the mass spectrometer and the controller.

## Typical Workflow

```python
from vimms.Chemicals import ChemicalMixtureCreator
from vimms.ChemicalSamplers import UniformMZFormulaSampler
from vimms.MassSpec import IndependentMassSpectrometer
from vimms.Controller import TopNController
from vimms.Environment import Environment

# 1. Generate chemicals
formula_sampler = UniformMZFormulaSampler(min_mz=100, max_mz=600)
cmc = ChemicalMixtureCreator(formula_sampler)
chemicals = cmc.sample(100, ms_levels=2)

# 2. Set up a virtual mass spectrometer
ms = IndependentMassSpectrometer(polarity="positive", chemicals=chemicals)

# 3. Choose a controller
controller = TopNController("positive", N=5, isolation_width=1)

# 4. Create and run the environment
env = Environment(ms, controller, min_time=0, max_time=1200)
env.run()
```

Running the environment produces a list of scans that can be written to mzML using `Environment.write_mzML()`. Evaluation data can also be stored by setting `save_eval=True` on the environment.

## IonClassifier Assignment Artifacts

`vimms.AssignmentChemicalArtifact` provides chemical-backed synthetic assignment
artifacts for IonClassifier-style local peak assignment problems:

```text
assignment artifact generator
  -> VIMMS chemicals
  -> optional IndependentMassSpectrometer + TopNController
  -> optional mzML/scan summary sidecars
  -> original truth-aligned picked feature table
```

The generated artifacts include candidate tables, picked-like feature tables,
chemical truth, assignment labels, and optional mzML/scan summaries. mzML
writing is disabled by default and should be enabled only for audit/export.
Assignment artifacts are materialized as chemicals so matched decoys,
interferents, blanks, and false MS2 support have a VIMMS chemical
representation. The model-facing peak table is not remeasured from scans: m/z,
RT, intensity, labels, and MS2 scores remain the original truth-aligned
assignment-table values. `UniformSpikeNoise` is not used for these artifacts
because spike peaks are not chemical-backed and cannot be fragmented.

The public assignment-noise APIs are `vimms.AssignmentNoise` for direct picked
tables and `vimms.AssignmentChemicalArtifact` for chemical-backed artifacts.
Implementation details for peak/artifact injection live in internal helper
modules.

### Adding Noise

The mass spectrometer accepts peak noise objects from `vimms.Noise` to make simulated spectra more realistic. For example:

```python
from vimms.Noise import GaussianPeakNoise
ms = IndependentMassSpectrometer("positive", chemicals, peak_noise=GaussianPeakNoise(0.01))
```

### Further Reading

* [Creating Chemicals](chemicals.md) explains how to generate chemical lists.
* [Running Controllers](controllers.md) describes available controllers and how to execute them.
* [Chromatographic Models](chromatography.md) details column offsets and drift.
* [Adding Noise](noise.md) shows how to generate more realistic spectra.
* [Evaluating Simulations](evaluation.md) covers metrics and analysis utilities.
* [Command Line Utilities](cli_tools.md) lists helper scripts.
