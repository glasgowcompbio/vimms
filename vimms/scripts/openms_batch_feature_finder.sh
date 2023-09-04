#!/bin/bash

# Get the directory path from the command-line argument
directory="$1"

# Check if the directory argument is provided
if [[ ! $directory ]]; then
  echo "Please provide the directory path."
  exit 1
fi

# Create an array to store the mzML files
mzml_files=()

# Function to process mzML files
process_mzml() {
  mzml_file="$1"
  remaining_files="$2"
  echo
  echo "Processing file: $mzml_file"

  # Print the remaining number of items
  echo "Remaining files: $remaining_files"

  # Execute the openms_feature_finder.sh script
  ./openms_feature_finder.sh "$mzml_file"

  echo "---------------------------------------------------"
}

# Recursive function to traverse the directory structure
traverse_directory() {
  local current_dir="$1"

  # Loop through the files and directories in the current directory
  for file in "$current_dir"/*; do
    if [[ -d "$file" ]]; then
      # If it's a directory, recursively call the function
      traverse_directory "$file"
    elif [[ -f "$file" && "$file" == *.mzML ]]; then
      # If it's a file with .mzML extension, add it to the mzml_files array
      mzml_files+=("$file")
    fi
  done
}

# Call the traverse_directory function with the specified directory
traverse_directory "$directory"

# Get the total number of files
total_files=${#mzml_files[@]}

# Process each mzML file
for ((i=0; i<total_files; i++)); do
  remaining_files=$((total_files - i))
  process_mzml "${mzml_files[i]}" "$remaining_files"
done
