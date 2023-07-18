#!/bin/zsh

# Assign the OpenMS bin directory to a variable
openms_dir="/Applications/OpenMS-2.8.0/bin"

# Get command line arguments
mzml_file=$1
ini_file=$2

# Check if mzML file is provided
if [[ ! $mzml_file ]]; then
    echo "Please specify the mzML file."
    exit 1
fi

# Check if the input mzML file exists
if [[ ! -e $mzml_file ]]; then
    echo "mzML file $mzml_file does not exist."
    exit 1
fi

# Check if ini file is provided, if not use default
if [[ ! $ini_file ]]; then
    ini_file="../../batch_files/FeatureFinderCentroided.ini"
fi

# Check if the ini file exists
if [[ ! -e $ini_file ]]; then
    echo "ini file $ini_file does not exist. Using default parameters."
    ini_file=""
fi

# Convert to CSV if the output file does not exist
output_csv="${mzml_file%.*}.csv"
if [[ -e $output_csv ]]; then
    echo "Output CSV file already exists. Skipping processing."
    echo "Process finished."
    exit 0
fi

# Create a temp directory if it doesn't exist
temp_dir="temp"
mkdir -p $temp_dir

# Run FeatureFinderCentroided
echo
echo "## Starting FeatureFinderCentroided using $ini_file"
temp_feature_featureXML="$temp_dir/temp_feature.featureXML"
$openms_dir/FeatureFinderCentroided -ini $ini_file -in $mzml_file -out $temp_feature_featureXML
echo "FeatureFinderCentroided finished. Output stored in $temp_feature_featureXML"

# Convert to CSV
echo
echo "## Converting to CSV"
$openms_dir/TextExporter -in $temp_feature_featureXML -out $output_csv
echo "Conversion to CSV finished. Output stored in $output_csv"

# Remove incorrect header lines from the CSV file
echo
echo "Removing incorrect header lines from the CSV file."
python remove_lines.py $output_csv
echo "Incorrect header lines removed."

# Clean up temporary files
echo
echo "## Cleaning up temporary files"
rm -rf $temp_dir
echo "Temporary files removed. Process finished."
