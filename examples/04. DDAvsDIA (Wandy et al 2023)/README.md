This folder contains notebooks that demonstrate how ViMMS can be to perform DDA vs DIA benchmarking,
both in the simulated case and on real experimental samples.

The folder 'simulation_experiments' contain two notebooks:
- `DDA_vs_DIA_run.ipynb` is the notebook to generate simulated runs to compare DDA vs DIA controllers
- `DDA_vs_DIA_matching.ipynb` is the notebook to analyse the results

The folder 'beer_analysis' contains notebooks to generate and analyse real experimental data:
- `1. Main DDA vs DIA results.ipynb` is the notebook to generate actual DDA and DIA LC-MS/MS runs using ViMMS. 
   Note that [Thermo IAPI](https://github.com/thermofisherlsms/iapi) license is required to run this notebook.
- `DDA_vs_DIA_real_peak_picking_25000.ipynb` is the notebook to match fullscan features to fragmentation spectra 
  for both DDA and DIA data, and also to construct the Multi-Injection reference library.
- `DDA_vs_DIA_real_similiarity_gnps_25000.ipynb` is the notebook to perform analysis using the GNPS/NIST14
  reference library.
- `DDA_vs_DIA_real_similiarity_intensity_non_overlap_25000.ipynb` is the notebook to perform analysis using
  the Multi-Injection reference library.