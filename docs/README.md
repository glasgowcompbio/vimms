---
nav_exclude: true
---
# Welcome to ViMMS

Welcome to the **V**irtual **M**etabolomics **M**ass **S**pectrometer (**VIMMS**), a comprehensive and modular framework for the simulation of fragmentation strategies in tandem mass spectrometry-based metabolomics. 

ViMMS allows you to simulate fragmentation strategies, generate virtual chemicals through various methods, and evaluate the performance of different strategies or controllers in either single or multi-sample settings. ViMMS is designed to serve as a unified platform for the development, testing, and optimization of fragmentation strategies in LC-MS metabolomics.

We also offer an extension that allows ViMMS controllers to operate directly on the [Thermo Orbitrap Fusion Tribrid](https://www.thermofisher.com/order/catalog/product/IQLAAEGAAPFADBMBCX) instrument. Please note that you'll need a license for [IAPI](https://github.com/thermofisherlsms/iapi) to use this feature.

# Installation

ViMMS is compatible with Python 3+. You can install the current release of ViMMS using pip:

```
$ pip install vimms
```

Find the current version on our [Release page](https://github.com/glasgowcompbio/vimms/releases) or on [PyPi](https://pypi.org/project/vimms/#history).

To use an older version like ViMMS 1.0, used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219), download it [here](https://zenodo.org/badge/latestdoi/196360601). However, note that this version may be outdated.

To access the latest, unreleased ViMMS code, clone our repository:

```
git clone https://github.com/glasgowcompbio/vimms.git
```

ViMMS dependencies are managed with [Poetry](https://python-poetry.org/). After cloning the repository, create a new virtual environment and install all required packages:
```
$ poetry install
$ poetry shell
```

Within the virtual environment, you can develop new controllers, run notebooks (`$ jupyter lab`), and more.

## Building the Documentation

ViMMS uses [MkDocs](https://www.mkdocs.org/) to build its documentation, which is hosted at [vimms.readthedocs.io](https://vimms.readthedocs.io). You can preview the
docs locally by running:

```bash
poetry run mkdocs serve
```

This command launches a local webserver so you can view the site at `http://127.0.0.1:8000/`.

## Running the Test Suite

If you plan on contributing, make sure all tests pass. Execute the following
from the project root:

```bash
./run_tests.sh
```

On Windows, use `run_tests.bat` instead. The test suite requires the
development dependencies installed via `poetry install`.

## More Information

Additional usage notes and installation instructions are available in our
[Installation guide](https://github.com/glasgowcompbio/vimms/blob/master/pages/installation.md),
while example notebooks can be found in the
[demo directory](https://github.com/glasgowcompbio/vimms/tree/master/demo).
