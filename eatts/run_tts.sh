#!/bin/bash

# Check if exactly three arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <python_path> <text> <emo>"
    exit 1
fi

# Assign arguments to variables
python_path=$1
text=$2
emo=$3

# Run the Python script with the provided parameters
#"$python_path" eatts.py "$text" "$emo"
