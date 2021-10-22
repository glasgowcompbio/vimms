---
title: 'ViMMS 2.0: A framework to develop, test and optimise fragmentation strategies in LC-MS metabolomics.' 
tags:
- Python
- mass spectrometry
- fragmentation
- simulation
- metabolomics 
authors:
  - name: Joe Wandy^[corresponding author]
    orcid: 0000-0002-3068-4664 
    affiliation: 1
  - name: Vinny Davies 
    ordic: 0000-0003-1896-8936 
    affiliation: 2
  - name: Ross McBride 
    affiliation: 3
  - name: Stefan Weidt 
    affiliation: 1
  - name: Simon Rogers 
    affiliation: 4
  - name: Rónán Daly
    affiliation: 1 
affiliations:
  - name: Glasgow Polyomics, University of Glasgow, Glasgow G12 8QQ, United Kingdom 
    index: 1
  - name: School of Mathematics and Statistics, University of Glasgow, Glasgow G12 8QQ, United Kingdom 
    index: 2
  - name: School of Computing Science, University of Glasgow, Glasgow G12 8QQ, United Kingdom 
    index: 3
  - name: NHS 
    index: 4 
date: 23 October 2021 
bibliography: paper.bib

---

# Summary

The choice of which fragmentation strategy to use for data acquisition directly affects the quality and coverage of
fragmentation data obtained from an untargeted metabolomics study, however developing novel fragmentation strategies is
challenging to high experimental cost and the lack of publicly available programmable simulator environment. ViMMS 2.0
is a framework that could be used to develop, test and optimise new fragmentation strategies completely in-silico. The
framework provides classes and methods to generate synthetic chemical objects and run them through an LC-MS/MS simulator
process. ViMMS also allows users to load previously generated data (in mzML format) and re-run them through different
fragmentation strategies. In this paper, we highlight the software design choices of the framework and illustrate with
examples the different scenarios of how ViMMS could be used to support the development of fragmentation strategies in
untargeted metabolomics.

# Statement of need

Metabolomics is the study of small molecules that participate in important cellular processes of an organism. Being the
closest to the phenotype, changes to metabolite levels are often expressed as a response to genetic or environmental
changes. Liquid chromatography (LC) coupled to mass spectrometry (MS/MS) is commonly used to identify small molecules in
untargeted experiments, where the identities of molecules of interests are not known in advance
[@smith2014proteomics]. In this setup, molecules elute through the LC column at different retention times (RTs)
before being isolated for fragmentation in the MS instrument. In tandem mass spectrometry, selected ions produced from
the survey scan (MS1) are further isolated for re-fragmentation in a second mass MS instrument (MS2), resulting in a
distinct spectral fingerprint for each molecule that could be used to improve identification.

Typically the raw LC-MS/MS measurements are processed in a data pre-processing pipeline to produce a list of
chromatographic peaks characterised by their m/z, RT and intensity values. During the identification process, molecular
annotations are assigned to peaks through matching with internal standard compounds (having known m/z and RT values) or
with spectral databases. An important factor that determines how many molecules could be matched to databases using
fragmentation data is the quality of the MS2 spectra during data acquisition. Good MS2 fragmentation strategies aims to
produce spectra for as many unknown ions in the sample as possible, but also produce high quality spectra which can be
reliably evaluated.

A common challenge faced by computational researchers with an interest in improving fragmentation strategies in LC-MS/MS
metabolomics is the lack of access and the high cost of MS instrument. This issue is particularly relevant as developing
and optimising novel fragmentation strategies tend to be conducted in an iterative fashion, where the method is
developed in the computer, validated on the instrument and optimised until the desired performance metric is reached. To
lower this barrier, we introduced Virtual Metabolomics Mass Spectrometer (
https://github.com/glasgowcompbio/vimms), a programmable and modular framework that simulates the chemical generation
process and the execution of fragmentation strategies in LC-MS/MS-based metabolomics.

# Related works

There are several existing mass spectrometry simulator in metabolomics, such as as JAMSS [@smith2015jamss],
Mspire-Simulator [@noyce2013mspire], MSAcquisitionSimulator [@goldfarb2016msacquisitionsimulator],
OpenMS-Simulator [@wang2015openms] and SMITER [@kosters2021smiter].

However none of these simulators are modular and programmable/scriptable in Python and work for metabolomics, which is
necessary to support the development of fragmentation strategies that respond to incoming scans in real time. Our work
ViMMS 1.0 [@Wandy2019-ok] was the first simulator that allowed for a simulator environment that could be controlled in
Python. However ViMMS 1.0 lacked several crucial aspects: its codebase was monolithic, making it difficult to
instantiate input from different sources, or to introduce different classes that extend the base functionality such as
introducing new classes to sample peak m/z, RT, intensity, chromatograms, or new fragmentation strategies.

![Overall ViMMS System Architecture, Data Flow, and Use Cases.\label{diagram}](figure.pdf)

# Improved ViMMS 2.0 Framework

ViMMS consists of two core modules: generation of chemicals from multiple sources, and running fragmentation strategies,
implemented as controller classes. An overview of the overall ViMMS architecture is given in \autoref{diagram}.

## Chemical Generations

Chemicals are input objects to the simulation process in ViMMS. Chemicals could be generated in many ways: either purely
synthetic or extracted from existing data (mzML files). The framework allows users to plug in modular classes that
specify parameters of chemicals, such as the distribution of their m/z values, RT, intensities, chromatographic shapes
and generated MS2 spectra, and also scan timing.

Chemicals could be generated in a single-sample or multi-sample settings. When generating multi-sample data, ViMMS
allows users to specify how chemicals could vary across samples. Given a list of base chemicals (chemicals that are
common) across samples, users could indicate the ratio of missing chemicals or how chemical intensities should vary in a
case-control setting.

## Running Fragmentation Strategies

Once chemical objects have been prepared whether for a single- or multi-sample setings, different fragmentation
strategies could be run. Fragmentation strategies are implemented as controllers. Controllers are executed in the
context of their environment, which put together the selected controller to run using the chemicals as input. We have
included a TopN fragmentation as the baseline controller. In addition, ViMMS 2.0 also includes implementations of
several enhanced controllers named SmartROI and WeightedDEW (outlined in [@davies21_smartroi]) which have been shown to
improve fragmentation coverage.

New controllers could be implemented by extending from our base controller. An example simple controller is described
here that randomly select N precursor ions for fragmentation.

```python
from vimms.Controller.base import Controller


class RandomController(Controller):  # TODO: finish this
    def __init__(self, params=None):
        super().__init__(params=params)

    def _process_scan(self, scan):
        new_tasks = []
        return new_tasks

    def update_state_after_scan(self, last_scan):
        pass

    def reset(self):
        pass
```

Something about evaluation here ...

## Software requirements

ViMMS 2.0 is distributed as a Python package that can be easily installed using pip. We require Python 3.0 or higher to
run ViMMS. It depends on common packages such as numpy, scipy and pandas. Automated unit tests is available in Python,
as well as continuous integration that build and run those unit tests. Our codebase is stored in Github and we welcome
contributions from interested people.

# Conclusion

in this paper, we have introduced ViMMS 2.0, an extension of the simulator framework in ViMMS 1.0, which is modular,
extensible and can be used in Python to simulate fragmentation strategies in untargeted metabolomics study. In other
works [@davies21_smartroi], the utility of ViMMS 2.0 have been validated through additional briding codes that allow
simulated controllers to run unchanged on both the simulator as well as on actual mass spectrometry. The proposed framework
could be used by different researchers to develop new and novel fragmentation strategies too.

# Acknowledgements

N/A.

# References
