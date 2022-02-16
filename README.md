# Welcome

Liquid-Chromatography (LC) coupled to tandem mass spectrometry (MS/MS) is widely used in identifying small molecules in
untargeted metabolomics. Various strategies exist to acquire MS/MS fragmentation spectra; however, the development of 
new acquisition strategies is hampered by the lack of framework that let researchers prototype, compare, and optimise 
strategies before validations on real machines. 

Here we introduce **V**irtual **M**etabolomics **M**ass **S**pectrometer 
(**VIMMS**), a programmable and modular framework that simulates fragmentation strategies in tandem mass spectrometry-based 
metabolomics. 

![ViMMS Logo](images/logo.png?raw=true "ViMMS Logo")

# Example Codes and Documentations

Can't wait to get started? Check out the following resources:
- Project documentation page: [![Documentation Status](https://readthedocs.org/projects/vimms/badge/?version=latest)](http://vimms.readthedocs.io/?badge=latest)
- Demo notebooks illustrating the use of the framework in a simulated setting can be found in the [Demo folder](https://github.com/glasgowcompbio/vimms/tree/master/demo).
- Example notebooks to accompany publications can be found in the [Example folder](https://github.com/glasgowcompbio/vimms/tree/master/examples)

# Features

ViMMS allows for scan-level control simulation of the MS2 acquisition 
process *in-silico*. It can be used to generate new LC-MS/MS data based on empirical data or virtually re-run a previous LC-MS/MS 
analysis using pre-existing data to allow the testing of different fragmentation strategies. Using ViMMS, we can 
compare different fragmentation strategies on real data, with the resulting scan results extractable as mzML files.

Additionally ViMMS can also be used as a framework to develop, optimise and test new fragmentation strategies. New 
fragmentation strategies can be implemented in ViMMS by extending from a Controller class. Such controllers can be run
on both the simulator as well as on actual mass spectrometry instruments that have supported APIs. 

### 1. Simulating fragmentation strategies

Popular fragmentation strategies, such as Top-N, can be implemented in ViMMS and 
tested on a wide variety of simulated data. The data can be generated completely in-silico
by sampling from a database of formulae, or generated with characteristics that resemble real data.
For example, please refer to [these notebooks](https://github.com/glasgowcompbio/vimms/tree/master/demo/01.%20Data).

### 2. Developing New Fragmentation Strategies

New fragmentation strategies can be developed on ViMMS by extending from a base Controller class. The controller can be run 
in either a simulated or a real environment (connecting to an actual mass spectrometry instrument). At the moment, we support
only the Thermo Fusion instrument via [IAPI](https://github.com/thermofisherlsms/iapi), but please contact us if you want to 
add support for other instruments.

- For example, here we show an example MS1 (fullscan) controller that only sends MS1 scans and saves them back: 
https://github.com/glasgowcompbio/vimms/blob/master/vimms/Controller/fullscan.py.

- Another example of a Top-N controller that fragments the top-N most intense precursor ions in the survey (MS1) scan:
https://github.com/glasgowcompbio/vimms/blob/master/vimms/Controller/topN.py.

# Installation

**Stable version**


ViMMS requires Python 3+. You can install the current release of ViMMS using pip or pipenv. 

```$ pip install vimms```
or
```$ pipenv install vimms```

The current version can be found in the [Release page](https://github.com/glasgowcompbio/vimms/releases).
or from [PyPi](https://pypi.org/project/vimms/#history).

**Older version**
- You can download ViMMS 1.0 used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219) from here: <a href="https://zenodo.org/badge/latestdoi/196360601"><img src="https://zenodo.org/badge/196360601.svg" alt="DOI"></a>
It contains codes up to the first paper, but they are quite out-of-date now. 

**Development version**

To use the latest bleeding-edge ViMMS code in this repository, follow the steps below to check out the master branch. Note that this repository is in active development, so some things may break (please report an issue in that case).

1. Clone this repository by checking out the master branch: `git clone https://github.com/glasgowcompbio/vimms.git`.
2. We provide two ways to manage the dependencies required by ViMMS. The first is using [Pipenv](https://pipenv.pypa.io/en/latest/), and the second is to use [Anaconda Python](https://www.anaconda.com). Refer to Section A and B below respectively.

***A. Managing Dependencies using Pipenv***

1. Install pipenv (https://pipenv.readthedocs.io).
2. In the cloned Github repo, run `$ pipenv install` to create a new virtual environment and install all the packages need to run ViMMS.
3. Go into the newly created virtual environment in step (4) by typing `$ pipenv shell`.
4. In this environment, you could develop new controllers, run notebooks (`$ jupyter lab`) etc. 

***B. Managing Dependencies using Pipenv***

1. Install Anaconda Python (https://www.anaconda.com/products/individual).
2. In the cloned Github repo, run `$ conda env create --file environment.yml` to create a new virtual environment and install all the packages need to run ViMMS.
3. Go into the newly created virtual environment in step (4) by typing `$ conda activate vimms`.
4. In this environment, you could develop new controllers, run notebooks (`$ jupyter lab`) etc. 

# Test Cases

![Vimms](https://github.com/glasgowcompbio/vimms/workflows/Vimms/badge.svg?branch=master&event=push)

Additionally unit tests that demonstrate how simulations can be run are available in the `tests` folder. You can use the script `run_tests.sh` or `run_tests.bat` to run them.

To run individual test classes you can use:

`python -m pytest <module>::<class>`

For example:

`python -m pytest tests/integration/test_controllers.py::TestSMARTROIController`

To see test output, add the `-s` switch, e.g.:

`python -m pytest -s tests/integration/test_controllers.py::TestSMARTROIController`

# Contributing

ViMMS is an MIT-licensed open-sourced project, and we welcome all contributions such as bugfixes, new features etc.
A guideline for community contribution can be found [here](https://github.com/glasgowcompbio/vimms/blob/master/CONTRIBUTING.md).

# Research

### Publications

To reference the ViMMS simulator in your work, please cite our paper:

[1] Wandy, Joe, et al. "*In Silico Optimization of Mass Spectrometry Fragmentation Strategies in Metabolomics.*" Metabolites 9.10 (2019): 219. https://www.mdpi.com/2218-1989/9/10/219

If you develop new controllers on top of ViMMS, please also cite the following paper: 

[2] Davies, Vinny, et al. "*Rapid Development of Improved Data-dependent Acquisition Strategies.*" Analytical Chemistry (2020). https://pubs.acs.org/doi/10.1021/acs.analchem.0c03895 ([data](http://researchdata.gla.ac.uk/1137/))

### Presentations

ViMMS has been presented in various metabolomics and computational biology venues. The following are our highlights:

##### Conference and Workshop Presentations 

- The following [poster](https://f1000research.com/posters/9-973) was presented in the CompMS COSI track in [ISMB 2020](https://www.iscb.org/ismb2020), and was voted the 
best poster for that track. An accompanying video presentation is also available. A similar poster was also presented
at [Metabolomics 2020 conference](http://metabolomics2020.org/).

[![SmartROI](http://img.youtube.com/vi/kHPYQicGoHE/0.jpg)](https://www.youtube.com/watch?v=kHPYQicGoHE "SmartROI")

- ViMMS was also presented as a talk at [Metabolomics 2021](https://www.metabolomics2021.org/) conference. You can find the [slides here](https://docs.google.com/presentation/d/e/2PACX-1vTADW9uJBYEMK91UGUw_99kHwn8jviT_Wvyj30Z2Akm0rswF_xbS_fUxuq23dVC4g/pub?start=false&loop=false&delayms=3000).

##### Departmental Talks

Slides and recorded videos from other departmental talks:
- [Computational Metabolomics as a Game of Battleships - Statistics Seminar, School of Mathematics and Statistics, University of Glasgow (2022)](https://media.ed.ac.uk/media/Vinny+Davies+%28University+of+Glasgow%29+Computational+Metabolomics+as+a+game+of+Battleships/1_as78pwks)
