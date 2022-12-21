This folder contains various notebooks to run experiments related to the TopNEXt framework.

These notebooks come in pairs:
Those suffixed with `_generation` use the ViMMS simulator to generate .mzML data in simulated experiments.
Those suffixed with `_evaluation` evaluate these results on metrics like coverage and produce visualisations.

* `simulated_multibeer` performs simulated experiments using a set of ten beers collected for the TopNEXt and DDA vs DIA experiments.
* `real_multibeer` provides the corresponding real experiments.
* `simulated_multibeer_replicate` executes a simulated replication study with 10 repeats of `simulated_multibeer` using different beers each time.
* `simulated_justinbeer` performs similar experiments to `simulated_multibeer` but using a different set of previously-published beers and urines.
* `real_multibeer_parameter` optimised the controller parameters for `simulated_multibeer` and `real_multibeer` on the Intensity Non-Overlap controller using a simulated experiment.
* `simulated_justinbeer_parameter` performs analogous parameter optimisations for `simulated_justinbeer` instead.
* `reoptimised_comparison` shows some of the results of `simulated_multibeer` but where parameters had been optimised for TopN Exclusion instead.