# Welcome to ViMMS

Welcome to **V**irtual **M**etabolomics **M**ass **S**pectrometer 
(**VIMMS**), a programmable and modular framework that simulates fragmentation strategies in tandem mass spectrometry-based 
metabolomics. 

You can use ViMMS to simulate various fragmentation strategies in a tandem mass spectrometry process. Virtual chemicals can
be generated by sampling either from HMDB, in a purely synthetic manner or extracted from existing mzML files
(through peak picking). These chemicals are then run through a simulated LC-MS process, and different
fragmentation strategies, implemented in the form of controllers, could be applied whether in a single- or multi-sample setting. 
The performance of different controllers can be evaluated using methods that we provide. 

ViMMS provides a unified framework to develop, test and optimise fragmentation strategies in LC-MS metabolomics.
An extension is also available to let ViMMS controllers to run directly on [Thermo Orbitrap Fusion Tribrid](https://www.thermofisher.com/order/catalog/product/IQLAAEGAAPFADBMBCX)
instrument. You'd need to have a license of [IAPI](https://github.com/thermofisherlsms/iapi) to do so -- please contact us if 
this is of any interest.

## Installation

### Stable version

ViMMS requires Python 3+. Unfortunately it is not compatible with Python 2. You can install a stable version 
of ViMMS using pip or pipenv. 

```$ pip install vimms```
or
```$ pipenv install vimms```

The current version can be found in the [Release page](https://github.com/glasgowcompbio/vimms/releases).
or from [PyPi](https://pypi.org/project/vimms/#history).

**Older version**
- You can download ViMMS 1.0 used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219) from here: <a href="https://zenodo.org/badge/latestdoi/196360601"><img src="https://zenodo.org/badge/196360601.svg" alt="DOI"></a>
It contains codes up to the first paper, but they are quite out-of-date now. 

### Development version

To use the latest bleeding-edge ViMMS code in this repository, follow the steps below to check out the master branch. Note that this repository is in active development, so some things may break (please report an issue in that case).

#### Using Pipenv

1. Install Python 3. We recommend Python >3.7.
2. Install pipenv (https://pipenv.readthedocs.io).
3. Clone this repository by checking out the master branch: `git clone https://github.com/glasgowcompbio/vimms.git`.
4. In this cloned directory, run `$ pipenv install` to create a new virtual environment and install all the packages need to run ViMMS.
5. Go into the newly created virtual environment in step (4) by typing `$ pipenv shell`.
6. Run Jupyter (`$ jupyter lab`) to see example notebooks.

#### Using Anaconda Python

1. Download the latest version of Anaconda Python from https://www.anaconda.com/products/individual.
2. Clone this repository by checking out the master branch: `git clone https://github.com/glasgowcompbio/vimms.git`.
3. In this cloned directory, run `$ conda env create -f environment.yml` to create a new virtual environment for Conda called 'vimms'.
4. Go into the newly created virtual environment in step (4) by typing `$ conda activate vimms`.
5Run Jupyter (`$ jupyter lab`) to see example notebooks.