---
title: Use Cases
nav_order: 4
---

# Use Cases for ViMMS

ViMMS (Virtual Metabolomics Mass Spectrometry) is a versatile framework designed for numerous applications in metabolomics. The tool's main use cases fall into three main categories: 
1. Simulating synthetic runs for data pre-processing pipeline testing.
2. Developing new data acquisition strategies.
3. Optimising parameters through resimulation of existing data.

Below we elaborate on each point.

## 1. Simulating Synthetic Runs for Pipeline Testing

Data pre-processing pipelines in metabolomics involve numerous operations including noise reduction, baseline correction, peak detection, and alignment. These pipelines require stringent testing to ensure reliable performance on real experimental data.

ViMMS addresses this need by simulating synthetic mzML files, which represent virtual runs of mass spectrometry experiments. The complexity and variability of these synthetic files mimic real-world data, providing an ideal testbed for assessing and optimising pre-processing pipelines. In this way, ViMMS contributes significantly to the refinement of data pre-processing stages, enhancing the overall quality of data analysis in metabolomics.

With ViMMS, you can implement and test widely used fragmentation strategies, such as Top-N, on various simulated data. This data can either be entirely generated in-silico, by sampling from a database of formulas, or produced to mirror the characteristics of real data. For an example, refer to [these notebooks](https://github.com/glasgowcompbio/vimms/tree/master/demo/01.%20Data).

To see a practical example, please refer to our publication:
- Wandy, Joe, et al. "[*Simulated-to-real Benchmarking of Acquisition Methods in Untargeted Metabolomics.*](https://www.frontiersin.org/articles/10.3389/fmolb.2023.1130781/full)" Frontiers in Molecular Biosciences 10 (2023): 1130781.

This study is a practical demonstration of ViMMS's capabilities in generating synthetic mzML files. It presents a comprehensive evaluation of Data-Dependent and Data-Independent Acquisition modes (DDA and DIA respectively), which are commonly employed for MS2 spectra acquisition in untargeted LC-MS/MS metabolomics analyses.

With ViMMS's ability to simulate varying parameters within both DDA and DIA modes, the study provides a systematic comparison of these methods without the need for extensive and costly real-world experiments. The findings indicate that the performance of the acquisition methods is influenced by the average number of co-eluting ions, an important factor in metabolomics analysis. This empirical evidence guides the appropriate choice of acquisition methods based on specific experimental conditions.

Importantly, the study substantiates its simulation-based results through experimental validation on an actual mass spectrometer. This confirmation highlights ViMMS's capacity to generate accurate and reliable simulated data that aligns with real-world observations. In conclusion, this study evidences ViMMS's potential in supporting rigorous data pre-processing pipeline testing and facilitating the development and optimisation of data acquisition methods in untargeted metabolomics.

## 2. Developing Novel Data Acquisition Strategies

Data acquisition in mass spectrometry directly influences the quality and comprehensiveness of data gathered from biological samples. Traditional acquisition methodologies often do not capture the full metabolic complexity of the samples, necessitating the development of improved techniques.

ViMMS facilitates the exploration of new data acquisition strategies through its flexible simulation environment. Users can adjust various mass spectrometry parameters, experimenting with different acquisition approaches in a controlled setting. This simulated environment enables the evaluation of innovative strategies, thereby advancing the development of data acquisition methodologies in metabolomics.

You can create new fragmentation strategies in ViMMS by extending from a base Controller class. These controllers can be run in both simulated and real-world settings, interfacing with actual mass spectrometry instruments. Currently, we only support the Thermo Fusion instrument via [IAPI](https://github.com/thermofisherlsms/iapi), but feel free to reach out if you want to add support for other instruments. For code examples, refer to this MS1 (fullscan) controller and a Top-N controller: [MS1 Controller](https://github.com/glasgowcompbio/vimms/blob/master/vimms/Controller/fullscan.py) and [Top-N Controller](https://github.com/glasgowcompbio/vimms/blob/master/vimms/Controller/topN.py).

The following works demonstrate how using ViMMS, novel data dependent acquisition (DDA) methods had been prototyped, developed and finally validated on the simulator and also deployed unchanged on the instrument.

- Davies, Vinny, et al. "[*Rapid Development of Improved Data-dependent Acquisition Strategies.*](https://pubs.acs.org/doi/10.1021/acs.analchem.0c03895)" Analytical Chemistry 19.43 (2021): 5676-5683.

The study by Davies et al. focuses on the development of new DDA methods using ViMMS. The authors first provide theoretical insights into the potential improvements achievable over current DDA strategies. Leveraging ViMMS as an in silico framework, they demonstrate the rapid and cost-efficient development of novel DDA methods. These methods incorporate advanced ion prioritisation strategies and are optimised through simulation-based experiments. The effectiveness of the developed methods is validated by fragmenting complex metabolite mixtures, showcasing their ability to fragment a higher number of unique ions compared to standard DDA strategies.

- Ross McBride, et al., [*TopNEXt: Automatic DDA Exclusion Framework for Multi-sample Mass Spectrometry Experiments*](https://academic.oup.com/bioinformatics/article/39/7/btad406/7207825), Bioinformatics, Volume 39, Issue 7, July 2023.

Similarly, McBride et al. present TopNEXt, a real-time scan prioritisation framework developed within the ViMMS framework. TopNEXt aims to enhance data acquisition in multi-sample liquid chromatography tandem mass spectrometry metabolomics experiments. By extending traditional Data-Dependent Acquisition exclusion methods across multiple samples, TopNEXt leverages a Region of Interest and intensity-based scoring system. Through simulated and laboratory experiments, the authors demonstrate that incorporating these novel concepts leads to the acquisition of fragmentation spectra for additional target peaks and with increased acquisition intensity. This improvement in the quality and quantity of fragmentation spectra holds potential for enhancing metabolite identification across various experimental contexts.

The availability and implementation of these novel DDA methods within the ViMMS framework underscore the platform's contribution to advancing data acquisition methodologies in metabolomics. Through ViMMS, researchers can explore, develop, and refine innovative strategies, paving the way for more comprehensive and reliable metabolomics analyses.

## 3. Resimulating Existing Data for Parameter Optimisation

ViMMS is not limited to synthetic data; it also enables the resimulation of real-world data. This functionality assists in determining optimal parameters for real-life mass spectrometry experiments.

The platform allows for the resimulation of actual experimental data under different parameter sets. By doing so, users can explore how various configurations influence their results. This iterative resimulation process can identify ideal experimental settings, leading to improved efficiency and accuracy in real experiments.

For an example, please refer to our work where we demonstrate how varying N and DEW (Dynamic Exclusion Window) could be explored in-silico, and the results validated on the actual instrument.

- Wandy, Joe, et al. "[*In Silico Optimization of Mass Spectrometry Fragmentation Strategies in Metabolomics.*](https://www.mdpi.com/2218-1989/9/10/219)" Metabolites 9.10 (2019): 219. 

In summary, ViMMS offers a comprehensive platform for metabolomics research. It enables the simulation, testing, and optimisation of pre-processing pipelines, data acquisition strategies, and experimental parameters. Its wide-ranging applications make it a valuable tool in the advancement of metabolomics, both in academic research and industry contexts.