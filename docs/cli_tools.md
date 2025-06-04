# Command Line Utilities

The `vimms/scripts` directory provides helper scripts for common tasks. They can be executed with Python or as standalone entry points once the package is installed.

## openms_optimise.py

Runs grid searches over controller parameters and evaluates each run with OpenMS. Results summarise coverage and intensity. Example usage:

```bash
python -m vimms.scripts.openms_optimise --mzml input.mzML --out-dir results
```

## openms_evaluate.py

Processes mzML output from a simulation (or real acquisition) to compute fragmentation coverage using OpenMS feature detection. The script produces a tabular report summarising how many peaks were fragmented.

## data_generation.py

Generates example chemical mixtures and mzML files, useful for testing pipelines or benchmarking fragmenters.

For a full listing of scripts refer to the [scripts folder](https://github.com/glasgowcompbio/vimms/tree/master/vimms/scripts).
