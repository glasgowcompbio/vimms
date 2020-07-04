Virtual Metabolomics Mass Spectrometry (ViMMS)
==============================================

Abstract
---------

![ViMMS Schematic](images/schematic.png?raw=true "ViMMS Schematic")

Liquid-Chromatography (LC) coupled to tandem mass spectrometry (MS/MS) is widely used in identifying small molecules in
untargeted metabolomics. Various strategies exist to acquire MS/MS fragmentation spectra; however, the development of 
new acquisition strategies is hampered by the lack of simulators that let researchers prototype, compare, and optimise 
strategies before validations on real machines. We introduce **V**irtual **M**etabolomics **M**ass **S**pectrometer 
(**VIMMS**), a modular metabolomics LC-MS/MS simulator framework that allows for scan-level control of the MS2 acquisition 
process *in-silico*. ViMMS can generate new LC-MS/MS data based on empirical data or virtually re-run a previous LC-MS/MS 
analysis using pre-existing data to allow the testing of different fragmentation strategies. It allows the 
comparison of different fragmentation strategies on real data, with the resulting scan results extractable as mzML files. 

To demonstrate its utility, we show how our proposed framework can be used to take the output of a real tandem mass 
spectrometry analysis and examine the effect of varying parameters in Top-N Data Dependent Acquisition protocol. 
We also demonstrate how ViMMS can be used to compare a recently published Data-set-Dependent Acquisition strategy with 
a standard Top-N strategy. We expect that ViMMS will save method development time by allowing for offline evaluation of 
novel fragmentation strategies and optimisation of the fragmentation strategy for a particular experiment.

Here is an example showing actual experimental spectra vs our simulated results.
![Example Spectra](images/spectra.png?raw=true "Example Spectra")

Installation
---------------

Stable version


ViMMS requires Python 3+. Unfortunately it is not compatible with Python 2. You can install the stable 1.0 version of ViMMS using pip:

```pip install vimms```

You can download the latest stable release (1.0) used in our paper from here: <a href="https://zenodo.org/badge/latestdoi/196360601"><img src="https://zenodo.org/badge/196360601.svg" alt="DOI"></a>

To use the latest bleeding-edge ViMMS code in this repository, follow the steps below. Note that this repository is in active development, so some things may break.

1. Install Python 3. We recommend Python 3.6 or 3.7.
2. Install pipenv (https://pipenv.readthedocs.io/en/latest/).
3. Clone this repository.
4. In this cloned directory, run `$ pipenv install` to create a new virtual environment and install all the packages need to run ViMMS.
5. Run Jupyter Notebook (`$ jupyter notebook`) or Jupyter Lab (`$ jupyter lab`).

Example Notebooks
--------

Notebooks that demonstrate how to use ViMMS are available in the [examples](https://github.com/sdrogers/vimms/tree/master/examples) folder of this repository.

Test Cases
--------

Additionally unit tests that demonstrate how simulations can be run are available in the `tests` folder. You can use the script `run_tests.sh` or `run_tests.bat` to run them.

Binder
--------

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sdrogers/vimms/master)


Publication
------------

To reference ViMMS in your work, please cite our paper:
- Wandy, Joe, et al. "*In Silico Optimization of Mass Spectrometry Fragmentation Strategies in Metabolomics.*" Metabolites 9.10 (2019): 219. https://www.mdpi.com/2218-1989/9/10/219
