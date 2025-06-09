# Untargeted Metabolomics Pipeline Plan

This document outlines the tasks for building a simple untargeted metabolomics
processing pipeline. Each task contains a short description and a status field to
track progress.

## Tasks

1. **Simulated Data Generation**
   - Create a Python script to generate 100 chemicals using `ChemicalMixtureCreator`.
   - Split chemicals into two groups (case/control) with 5 samples each.
   - Use reasonable metabolomics ranges (e.g. m/z 100--1000, RT <3 min) with
     chromatograms roughly 5 s wide (``sigma≈1``).
   - Acquire data with Top‑1 DDA in positive mode including common adducts.
   - Export mzML files, a ground‑truth table linking peaks to compounds, and a
     simple MGF library of MS2 spectra.
   - **Status:** Completed (mzML, ground truth table, and MGF library generated)

2. **Peak Picking**
   - Since the chemicals are simulated, derive peak information directly from the
     ground truth and produce a table of peaks per sample (bounding box, apex m/z,
     RT, intensity, sample name, etc.).
   - **Status:** Not started

3. **Alignment**
   - Implement a simple join aligner that merges peaks across samples to create
     an aligned feature table. Optionally use MZmine for comparison.
   - **Status:** Not started

4. **Grouping Related Peaks**
   - Cluster related peaks (e.g. isotopes/adducts) using a simple RT clustering
     approach such as a Dirichlet process mixture model. Provide an option to
     run a similar step in MZmine.
   - **Status:** Not started

5. **Identification**
   - Match observed peaks to the ground‑truth MSP library to assess performance.
   - **Status:** Not started

A main pipeline script will orchestrate these steps once they are implemented.
