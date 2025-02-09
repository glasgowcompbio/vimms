This folder contains notebooks for experiments related to the matching methods.

- `simulated_multibeer_matching_generation` runs the main experiments. `simulated_multibeer_matching_evaluation` produces evaluations of those experiments.
- `peak_picking` runs XCMS or MZMine separately from the above notebooks, to make the process a bit more systematic and avoid redundantly re-running the peak-picker between experiments.
- `dsda_optimisation` finds the best set of parameters to run DsDA with for the experiments.
- `resync_generation` and `resync_evaluation` test the performance of the pre-planned methods when scan times consistently deviated by those expected by the methods.