# Installation Guide

**Stable Release**

ViMMS is compatible with Python 3 or newer. You can install the latest stable release using pip or pipenv:

```$ pip install vimms```
or
```$ pipenv install vimms```

To verify the current version, visit the [Release page](https://github.com/glasgowcompbio/vimms/releases) or [PyPi](https://pypi.org/project/vimms/#history).

**Older Release**

The ViMMS 1.0 version used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219) can be downloaded [here](https://zenodo.org/badge/latestdoi/196360601). Please note, this version is now significantly out of date.

**Development Version**

For the most recent codebase (still under development), clone this repository:

```$ git clone https://github.com/glasgowcompbio/vimms.git```

The dependencies can be managed using either [Pipenv](https://pipenv.pypa.io/en/latest/) or [Anaconda Python](https://www.anaconda.com).

***With Pipenv:***

1. Install pipenv.
2. Run `$ pipenv install` within the cloned repo to create a new virtual environment and install required packages.
3. Enter the virtual environment using `$ pipenv shell`.

***With Anaconda Python:***

1. Install Anaconda Python.
2. Run `$ conda env create --file environment.yml` within the cloned repo to create a new virtual environment and install required packages.
3. Access the virtual environment by typing `$ conda activate vimms`.

# Test Cases

Unit tests demonstrating simulation execution are in the `tests` folder. Use scripts `run_tests.sh` or `run_tests.bat` to run these tests.

Run individual test classes with:

```$ python -m pytest <module>::<class>```

For example:

```$ python -m pytest tests/integration/test_controllers.py::TestSMARTROIController```

Include `-s` switch for test output:

```$ python -m pytest -s tests/integration/test_controllers.py::TestSMARTROIController```