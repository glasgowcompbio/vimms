from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    dependency_links=[],
    name="vimms",
    version="2.0.6",
    author="Joe Wandy, Vinny Davies, Ross McBride, Justin J.J. van der Hooft, "
           "Stefan Weidt, Ronan Daly, Simon Rogers",
    author_email="joe.wandy@glasgow.ac.uk",
    description="A framework to develop, test and optimise fragmentation strategies in LC-MS "
                "metabolomics.",
    long_description="ViMMS is a modular LC-MS/MS simulator framework for "
                     "metabolomics that allows for real-time scan-level "
                     "control of the MS2 acquisition process in-silico.",
    long_description_content_type="text/markdown",
    url="https://github.com/glasgowcompbio/vimms",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'scikit-learn',
                      'pymzml==2.4.7', 'psims', 'events', 'tqdm', 'joblib', 'ipyparallel',
                      'requests', 'loguru', 'networkx', 'jsonpickle', 'statsmodels',
                      'mass-spec-utils', 'tabulate', 'pysmiles', 'intervaltree', 'gpy'],

)
