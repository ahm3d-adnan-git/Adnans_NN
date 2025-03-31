#!/bin/bash

# Echo a message to indicate what the script is supposed to do
echo "This script should trigger the training and testing of your neural network implementation..."

# Function to log detailed information
log_info() {
    echo "[INFO] $1"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    log_info "Usage: $0 <input_config>"
    exit 1
fi

# Assign the first argument to CONFIG_FILE
CONFIG_FILE=$1

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    log_info "Error: Config file '$CONFIG_FILE' does not exist."
    exit 1
fi

# Compile main.cpp if the executable does not exist
EXECUTABLE="./read_config"
if [ ! -f "$EXECUTABLE" ]; then
    g++ -fopenmp -O2 -o read_config src/main.cpp -std=c++20
    if [ $? -ne 0 ]; then
        log_info "Compilation failed. Please fix errors in main.cpp."
        exit 1
    fi
fi

# Run the executable with the configuration file
$EXECUTABLE "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    log_info "Error: Failed to process the config."
    exit 1
fi
