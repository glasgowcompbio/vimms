from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    dependency_links=[],
    name="vimms",
    version="2.0.3",
    author="Joe Wandy, Vinny Davies, Justin J.J. van der Hooft, Stefan Weidt, Ronan Daly, Simon Rogers",
    author_email="joe.wandy@glasgow.ac.uk",
    description="ViMMS is an LC-MS/MS simulator.",
    long_description="ViMMS is a modular LC-MS/MS simulator framework for metabolomics that allows for real-time scan-level control of the MS2 acquisition process in-silico. ViMMS allows users to simulate and test different fragmentation strategies and obtain fragmentation files in .mzML format as output from the simulation (the entire state of the simulator can also be saved for inspection later).",
    long_description_content_type="text/markdown",
    url="https://github.com/sdrogers/vimms",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'pandas', 'seaborn', 'scikit-learn', 'matplotlib', 'pymzml==2.4.7',
                      'psims', 'events', 'tqdm', 'joblib', 'ipyparallel', 'requests', 'loguru', 'networkx', 'jsonpickle',
                      'statsmodels', 'mass-spec-utils', 'tabulate', 'pysmiles', 'pipenv-setup', 'flake8', 'pytest',
                      'intervaltree', 'jupyterlab', 'gpy'],

)
