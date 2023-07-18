import sys

# Get the filename from command line argument
filename = sys.argv[1]

# Read the file contents
with open(filename, 'r') as file:
    lines = file.readlines()

# Remove lines 1, 2, 3, and 5 as they aren't the correct headers
# i.e. the correct header is in line 4 ...
lines_to_remove = [1, 2, 3, 5]
filtered_lines = [line for i, line in enumerate(lines) if i + 1 not in lines_to_remove]

# Write the modified contents back to the file
with open(filename, 'w') as file:
    file.writelines(filtered_lines)