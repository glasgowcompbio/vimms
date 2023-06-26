#!/bin/bash

in_mzml="/home/joewandy/data/BSA_100fmol__recon_1ul_1.mzML"
at_least_one_point_above="1E4"
source_dir="/home/joewandy/vimms/vimms/scripts/topN_timing_improvement_1E4"

# An array of charge range start and end
# charge_range=( "1 8" "2 6" )
charge_range=( "2 6" )

# An array of min_averagine_scores
# min_averagine_scores=( "50" "100" "150" "200" )
min_averagine_scores=( "160" "170" "180" "190" )

# Loop through each combination of charge range and min_averagine_scores
for range in "${charge_range[@]}"; do
    IFS=' ' read -r -a tokens <<< "$range"
    start=${tokens[0]}
    end=${tokens[1]}
    for score in "${min_averagine_scores[@]}"; do
        out_dir="topN_timing_improvement_${at_least_one_point_above}_${start}_${end}_${score}"
        # Check if directory exists, if not create it
        if [ ! -d "$out_dir" ]; then
          mkdir -p $out_dir
          # Copy contents of source directory to new directory
          cp -r $source_dir/* $out_dir/
        fi
        python topN_test.py --in_mzml $in_mzml --at_least_one_point_above $at_least_one_point_above --charge_range_start $start --charge_range_end $end --out_dir $out_dir --min_averagine_score $score
    done
done