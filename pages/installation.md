# ViMMS Installation Guide

## 🌟 Stable Release
ViMMS is designed for Python 3 and above. Install the latest stable release with the following command:

```bash
pip install vimms
```
Check out the latest versions on the [Release page](https://github.com/glasgowcompbio/vimms/releases) or [PyPi](https://pypi.org/project/vimms/#history).

**🕰 Older Releases**

For those interested in ViMMS version 1.0 as used in our [original paper](https://www.mdpi.com/2218-1989/9/10/219), you can get it [here](https://zenodo.org/badge/latestdoi/196360601). 
Be aware that it's quite outdated now.
For other previous releases, head over to the [Releases](https://github.com/glasgowcompbio/vimms/releases) page on GitHub. 
This include releases to support other papers.

**🔧 Development Version**

To get the latest features and fixes (still under development), clone the repository:

```$ git clone https://github.com/glasgowcompbio/vimms.git```

You can then set up the environment using [Anaconda Python](https://www.anaconda.com) or [Poetry](https://python-poetry.org).
We recommend using Conda.

There is also support for using [Pipenv](https://pipenv.pypa.io/en/latest/) through the included Pipfile in the repo, but 
going forward that will not be maintained anymore.

***Setting up with Anaconda:***
```
$ cd vimms
$ conda env create --file environment.yml
$ conda activate vimms
$ jupyter lab (to test notebooks)
```

***Setting up with Poetry:***
```
$ cd vimms
$ pip install poetry (if you don't have it)
$ poetry install
$ poetry shell
$ jupyter lab (to test notebooks)
```

***Setting up with Pipenv:***
```
$ cd vimms
$ pip install pipenv (if you don't have it)
$ pipenv install
$ pipenv shell
```

# 🧪 Testing ViMMS

Unit tests are located in the `tests` folder. Use the scripts `run_tests.sh` or `run_tests.bat` to execute them.

Run individual test classes with:

```$ python -m pytest <module>::<class>```

For example:

```$ python -m pytest tests/integration/test_controllers.py::TestSMARTROIController```

Include `-s` switch for test output:

```$ python -m pytest -s tests/integration/test_controllers.py::TestSMARTROIController```