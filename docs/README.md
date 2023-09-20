---
nav_exclude: true
---
# Welcome to ViMMS

Welcome to the **V**irtual **M**etabolomics **M**ass **S**pectrometer (**VIMMS**), a comprehensive and modular framework for the simulation of fragmentation strategies in tandem mass spectrometry-based metabolomics. 

ViMMS allows you to simulate fragmentation strategies, generate virtual chemicals through various methods, and evaluate the performance of different strategies or controllers in either single or multi-sample settings. ViMMS is designed to serve as a unified platform for the development, testing, and optimization of fragmentation strategies in LC-MS metabolomics.

We also offer an extension that allows ViMMS controllers to operate directly on the [Thermo Orbitrap Fusion Tribrid](https://www.thermofisher.com/order/catalog/product/IQLAAEGAAPFADBMBCX) instrument. Please note that you'll need a license for [IAPI](https://github.com/thermofisherlsms/iapi) to use this feature.

# Installation

ViMMS is compatible with Python 3+. You can install the current release of ViMMS using pip or pipenv:

```
$ pip install vimms
```
or
```
$ pipenv install vimms
```

Find the current version on our [Release page](https://github.com/glasgowcompbio/vimms/releases) or on [PyPi](https://pypi.org/project/vimms/#history).

To use an older version like ViMMS 1.0, used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219), download it [here](https://zenodo.org/badge/latestdoi/196360601). However, note that this version may be outdated.

To access the latest, unreleased ViMMS code, clone our repository:

```
git clone https://github.com/glasgowcompbio/vimms.git
```

ViMMS dependencies can be managed using either [Pipenv](https://pipenv.pypa.io/en/latest/) or [Anaconda Python](https://www.anaconda.com). After installing your chosen tool and cloning the repo, create a new virtual environment and install all required packages:

For Pipenv:
```
$ pipenv install
$ pipenv shell
```

For Anaconda:
```
$ conda env create --file environment.yml
$ conda activate vimms
```

Within the virtual environment, you can develop new controllers, run notebooks (`$ jupyter lab`), and more.