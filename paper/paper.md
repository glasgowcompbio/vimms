---
title: 'ViMMS 2.0: A framework to develop, test and optimise fragmentation strategies in LC-MS metabolomics' 
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
  - name: Glasgow Polyomics, University of Glasgow, United Kingdom 
    index: 1
  - name: School of Mathematics and Statistics, University of Glasgow, United Kingdom
    index: 2
  - name: School of Computing Science, University of Glasgow, United Kingdom 
    index: 3
  - name: NHS National Services Scotland, United Kingdom
    index: 4 
date: 26 October 2021 
bibliography: paper.bib

---

# Summary

The choice of fragmentation strategies used during data acquisition directly affects the quality and coverage during
identification -- a crucial step in untargeted metabolomics data analysis. However, developing novel fragmentation
strategies is challenging due to high experimental cost of running an actual mass spectrometry instrument and the lack
of programmable simulation environment to support developmental activities. ViMMS 2.0 is a framework that could be used
to develop new fragmentation strategies in metabolomics completely *in-silico* and also to run them on actual mass
spectrometry instruments. The framework allows users to generate chemical objects (produced synthetically or extracted
from existing mzML files) and simulate a tandem mass spectrometry process, where different fragmentation strategies
could rapidly be implemented, tested and evaluated. In this paper, we highlight the software design choices of the
framework and illustrate with an example how a new fragmentation strategy could be implemented in ViMMS 2.0.

# Statement of need

Metabolomics is the study of small molecules that participate in important cellular processes of an organism. Being the
closest to the phenotype, changes to metabolite levels are often expressed as a response to genetic or environmental
changes [@guijas2018metabolomics]. Liquid chromatography (LC) coupled to mass spectrometry (MS) is commonly used to
identify small molecules in untargeted experiments, where the identities of molecules of interests are not known in
advance
[@smith2014proteomics]. In this setup, molecules elute through the LC column at different retention times (RTs)
before being isolated for fragmentation in the MS instrument. In tandem mass spectrometry, selected ions produced from
the survey scan (MS1) are further isolated for re-fragmentation in a second MS instrument (MS2), resulting in a distinct
spectral fingerprint for each molecule that could be used to improve identification.

Typically the raw LC-MS/MS measurements are processed in a data pre-processing pipeline to produce a list of
chromatographic peaks characterised by their m/z, RT and intensity values. During identification, molecular annotations
are assigned to peaks through matching with internal standard compounds (having known m/z and RT values) or with
spectral databases. An important factor that determines how many molecules could be matched and annotated to databases
using fragmentation data is the quality of the MS2 spectra during data acquisition. Good MS2 fragmentation strategies
aims to produce spectra for as many unknown ions in the sample as possible, but also produce high quality spectra which
can be reliably evaluated.

A common challenge faced by computational researchers with an interest in improving fragmentation strategies in tandem
mass-spectrometry-based metabolomics is the lack of access and the high cost of running MS instrument. This issue is
particularly relevant as developing and optimising novel fragmentation strategies tend to be conducted in an iterative
fashion, where the method is developed in the computer, validated on the instrument and optimised until the desired
performance metric is reached. To lower this barrier, we introduced **Vi**rtual **M**etabolomics **M**ass **S**
pectrometer (ViMMS) 2.0, a programmable and modular framework that simulates the chemical generation process and the
execution of fragmentation strategies in LC-MS/MS-based metabolomics.

# Related works

Existing mass spectrometry simulators are ill-fitted to support rapid development of fragmentation strategies that
respond to incoming scans in real time. This is primarily due to the limited ways users could incoporate new strategies
within existing simulators' codebase. Currently available simulators such as as Mspire-Simulator [@noyce2013mspire],
JAMSS [@smith2015jamss], OpenMS-Simulator [@wang2015openms],
MSAcquisitionSimulator [@goldfarb2016msacquisitionsimulator] and SMITER [@kosters2021smiter] exist as stand-alone
programs or GUI applications, and are not easily scriptable or programmable.

Additionally the above-highlighted simulators operate on proteomics data, with proteins as the objects of interest. In
principle they could be extended to support metabolomics data, but this is not a trivial change. Our work in ViMMS
1.0 [@Wandy2019-ok] was the first simulator that allowed for a metabolomics-based simulation environment that could
easily extended in Python. However ViMMS 1.0 suffered from several weaknesses: its codebase was monolithic, making it
difficult to instantiate input from different sources or to introduce different extensions to the base functionalities,
including adding new fragmentation strategies. The focus of ViMMS 1.0 was more on simulating a complete tandem mass
spectrometry run in metabolomics, and not as much as in enabling the development of new strategies.

# Improved ViMMS 2.0 Framework

In ViMMS 2.0 we significantly improved the simulator framework architecture with the goal of supporting fragmentation
strategies development and validation. An overview of the overall architecture is given in \autoref{diagram}. The
simulator framework consists of two core functionalities: the generation of chemicals from multiple sources, and
executions of fragmentation strategies, implemented as controller classes. The improved modularity in ViMMS 2.0 allows
many aspects of the framework to be swapped out with alternative implementations, including classes that generate
various aspects of chemicals for simulation (\autoref{diagram}A), mass spectrometry simulator (\autoref{diagram}B),
specific controllers that implement various fragmentation strategies (\autoref{diagram}C), as well as the environmental
context to run them all (\autoref{diagram}D).

![Overall ViMMS System Architecture.\label{diagram}](figure.pdf)

## Generating Input Chemicals for Simulation

Chemicals are input objects to the simulation process in ViMMS 2.0, and could be generated in many ways: either in a
purely synthetic manner or extracted from existing data (mzML files) via peak picking. The framework allows users to
plug in modular classes that specify parameters of chemicals, such as the distribution of their m/z values, RT,
intensities, chromatographic shapes and generated MS2 spectra, as well as scan timing (\autoref{diagram}A).

Chemicals could be generated in a single-sample or multi-sample settings. When generating multi-sample data, ViMMS
allows users to specify how chemicals could vary across samples. Given a list of base chemicals (chemicals that are
common) across samples, users could indicate the ratio of missing chemicals or how chemical intensities should vary in a
case-control setting.

## Implementing and Running Fragmentation Strategies

Once chemical objects have been prepared (whether for a single- or multi-sample settings), different fragmentation
strategies could be run. Fragmentation strategies are implemented as controllers that extend from the base `Controller`
class. Controllers are executed in the context of their environment, which brings together input chemicals, mass
spectrometry and controllers in a single context (\autoref{diagram}D). Note that the modularity of the mass spectetry
and environment means it is possible to swap purely simulated MS and environment implementation with alternatives that
control an actual MS instrument. In another work, we demonstrated the practicality of this idea by building alternative
implementations of these classes that allow fragmentation strategies to be executed unchanged both in simulation as well
as on Thermo Tribrid Fusion instrument [@davies21_smartroi].

To illustrate how new strategies could be built on top of ViMMS 2.0, an example is given here of a Top-N controller that
select the *N* most intense precursor ions for fragmentation in a typical data-dependant acquisition (DDA) fashion. In
this implementation, the controller `SimpleTopNController` extends from a base `Controller` that provide base methods to
handles various scan interactions with the mass spectrometry. The implementation has to override the `_process_scan`
method, which determines how precursor ions in a newly received MS1 scan are prioritised to generate further
fragmentation scans. Although it is not shown in this example, other methods in the parent `Controller` could also be
overriden for different purposes, such as responding when an acquisition has been started or stopped.

```python
import numpy as np
from vimms.Controller.base import Controller


class SimpleTopNController(Controller):

    def _process_scan(self, scan):
        new_tasks = []

        # If the controller has received an MS1 scan to process        
        if self.scan_to_process is not None:

            # Extract m/z and intensity values in this MS1 scan
            mzs = self.scan_to_process.mzs
            intensities = self.scan_to_process.intensities

            # Select only the Top-N precursors, sorted by intensities descending
            idx = np.argsort(intensities)[::-1]
            idx = idx[0:self.N]

            # Loop over the Top-N precursors and target them for fragmentation
            for i in idx:
                mz, intensity = mzs[i], intensities[i]

                # Schedule a new MS2 scan targeting the selected precursor ion
                dda_scan_params = self.get_ms2_scan_params(mz, intensity, ...)
                new_tasks.append(dda_scan_params)
                self.current_task_id += 1

            # Schedule the next survey MS1 scan after doing N MS2 scans
            ms1_scan_params = self.get_ms1_scan_params()
            new_tasks.append(ms1_scan_params)
            self.current_task_id += 1

            # Set this MS1 scan as has been processed
            # and indicate what is the next MS1 scan id to process
            self.scan_to_process = None
            self.next_processed_scan_id = self.current_task_id

        # Return all the scheduled tasks to be executed by the mass spec
        return new_tasks
```

The simple Top-N scheme above could be enhanced to incorporate dynamic exclusion windows to prevent the same precursors
from being fragmented repeatedly, or to incorporate different schemes of prioritising which precursor ions to fragment.
We have included a more complete Top-N fragmentation as the baseline controller in ViMMS 2.0 against which other
strategies can be compared to. Two enhanced DDA controllers (named **SmartROI** and **WeightedDEW**, outlined
in [@davies21_smartroi]) are also provided that demonstrate how novel fragmentation strategies could be rapidly
implemented and validated in ViMMS 2.0. SmartROI accomplishes this by tracking regions-of-interests in real-time and
targeting those for fragmentations, while WeightedDEW performs a weighted dynamic exclusion schemes to prioritise
precursor ions for fragmentations. Evaluation codes are provided to compute various evaluation metrics, such as coverage
and fragmented precursor intensities. This allows users to benchmark different controller implementations in a
comparative setting.

## Software requirements

ViMMS 2.0 is distributed as a Python package that can be easily installed using pip. We require Python 3.0 or higher to
run the framework. It depends on common packages such as numpy, scipy and pandas. Automated unit tests is available in
Python, as well as continuous integration that build and run those unit tests in our code repository. Our codebase is
stored in Github and we welcome contributions from researchers with interest in developing novel fragmentation
strategies in both data-dependant and data-independant acquisitions.

# Conclusion

in this paper, we have introduced ViMMS 2.0, an extension of the simulator framework in ViMMS 1.0, which is modular,
extensible and can be used in Python to simulate fragmentation strategies in untargeted metabolomics study. In other
works [@davies21_smartroi], the utility of ViMMS 2.0 have been validated through additional briding codes that allow
simulated controllers to run unchanged on both the simulator as well as on actual mass spectrometry. It is our hope that
the work outlined here would be used to advance the development of novel fragmentation strategies in untargeted
metabolomics.

# Acknowledgements

N/A.

# References
