#!/bin/bash

in_mzml="/home/joewandy/data/HELA_20ng_1ul__sol_3.mzML"
at_least_one_point_above="1E4"
source_dir="/home/joewandy/vimms/vimms/scripts/hela_1E4"

# Base directory for all output
base_out_dir="hela_results"

# Variables for charge range start and end
charge_range_start="2"
charge_range_end="3"

# An array of min_fit_scores and penalty factors
min_fit_scores=( "20" "40" "60" "80" "100" "120" "140" "160" "180" "200" )

# An array of penalty factors
penalty_factors=( "0.25" "0.50" "0.75" "1.0" "1.25" "1.50" "1.75" "2.0" )

# Check if the parallel option is specified
if [ "$1" == "--parallel" ]; then
    parallel=true
else
    parallel=false
fi

# Check if base directory exists, if not create it
if [ ! -d "$base_out_dir" ]; then
  mkdir -p $base_out_dir
fi

# Loop through each combination of min_fit_scores and penalty_factors
job_count=0
for score in "${min_fit_scores[@]}"; do
    for penalty in "${penalty_factors[@]}"; do
        out_dir="${base_out_dir}/hela_${at_least_one_point_above}_${charge_range_start}_${charge_range_end}_${score}_${penalty}"
        # Check if directory exists, if not create it
        if [ ! -d "$out_dir" ]; then
          mkdir -p $out_dir
          # Copy contents of source directory to new directory
          cp -r $source_dir/* $out_dir/
        fi
        # Run the script in the background if --parallel is specified
        if [ "$parallel" = true ]; then
            python topN_test.py --in_mzml $in_mzml --at_least_one_point_above $at_least_one_point_above --charge_range_start $charge_range_start --charge_range_end $charge_range_end --out_dir $out_dir --min_fit_score $score --penalty_factor $penalty  --use_quick_charge &
            ((job_count++))
            # If we've reached 10 jobs, wait for any job to complete
            if (( job_count % 10 == 0 )); then
                wait -n
            fi
        else
            python topN_test.py --in_mzml $in_mzml --at_least_one_point_above $at_least_one_point_above --charge_range_start $charge_range_start --charge_range_end $charge_range_end --out_dir $out_dir --min_fit_score $score --penalty_factor $penalty --use_quick_charge
        fi
    done
done

# If --parallel is specified, wait for all background jobs to finish
if [ "$parallel" = true ]; then
    wait
fi
