![ViMMS Logo](images/logo.png?raw=true "ViMMS Logo")

# About

Liquid-Chromatography (LC) coupled with tandem mass spectrometry (MS/MS) is a prevalent technique for identifying small molecules in untargeted metabolomics. There are various strategies for acquiring MS/MS fragmentation spectra, but devising new methods is often challenging due to the absence of a structured environment where researchers can prototype, compare, and optimize strategies before testing on real equipment. 

To solve this, we introduce the **V**irtual **M**etabolomics **M**ass **S**pectrometer (**VIMMS**), a flexible and modular framework designed to simulate fragmentation strategies in tandem mass spectrometry-based metabolomics. 

# Quick Start & Documentation

Eager to start using ViMMS? Take advantage of these resources:
- [Installation guide](pages/installation.md).
- Visit our project documentation page: [![Documentation Status](https://readthedocs.org/projects/vimms/badge/?version=latest)](http://vimms.readthedocs.io/?badge=latest)
- Our [Demo folder](https://github.com/glasgowcompbio/vimms/tree/master/demo) contains notebooks that demonstrate how to use the framework in a simulated environment.
- For specific examples that accompany our publications, see the [Example folder](https://github.com/glasgowcompbio/vimms/tree/master/examples).
- You can also find this [quick guide on how to get started using ViMMS](https://github.com/glasgowcompbio/vimms/blob/master/demo/guide_to_vimms/guide_to_vimms.ipynb).

# Key Features

ViMMS provides scan-level control simulation of the MS2 acquisition process in a virtual environment. You can generate new LC-MS/MS data based on empirical data or virtually replay a previous LC-MS/MS analysis using existing data, which allows for testing different fragmentation strategies. With ViMMS, you can evaluate diverse fragmentation strategies using real data, and extract the scan results as mzML files.

Moreover, ViMMS serves as a platform for the development, optimization, and testing of new fragmentation strategies. These strategies can be implemented by extending a Controller class in ViMMS, and can be tested on both the simulator and actual mass spectrometry instruments that support compatible APIs.

To see a more thorough explanation of the use cases of ViMMS, please refer to the [Use Cases](pages/use_cases.md) section.


# Contributions

As an open-source project licensed under MIT, we welcomes all forms of contributions, including bug fixes, new features, and more. You can find our community contribution guidelines [here](https://github.com/glasgowcompbio/vimms/blob/master/CONTRIBUTING.md).

# Citing ViMMS 

To cite ViMMS or read about the list of publications that are built on top of ViMMS, please refer to the [Publications](pages/publications.md) page.
ViMMS is also actively [presented](pages/talks.md) in various computational biology venues.
