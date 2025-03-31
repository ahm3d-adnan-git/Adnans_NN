#!/bin/bash

echo "This script should read a dataset image into a tensor and pretty-print it into a text file..."

# Function to log detailed info
log_info() {
    echo "[INFO] $1"
}

# Check for required arguments
if [ "$#" -ne 3 ]; then
    log_info "Usage: $0 <input_file> <output_file> <image_index>"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2
IMAGE_INDEX=$3


# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    log_info "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

# Check if the image index is a valid number
if ! [[ "$IMAGE_INDEX" =~ ^[0-9]+$ ]]; then
    log_info "Error: Image index must be a non-negative integer."
    exit 1
fi

# Compile the main.cpp if needed
EXECUTABLE="./read_image"
if [ ! -f "$EXECUTABLE" ]; then
    g++ -fopenmp -O2 -o read_image src/main.cpp -std=c++20
    if [ $? -ne 0 ]; then
        log_info "Compilation failed. Please fix errors in main.cpp."
        exit 1
    fi
fi


# Run the executable
# log_info "Running the program..."
$EXECUTABLE "$INPUT_FILE" "$OUTPUT_FILE" "$IMAGE_INDEX"
if [ $? -ne 0 ]; then
    log_info "Error: Failed to process the image."
    exit 1
fi


# log_info "Image $IMAGE_INDEX written to $OUTPUT_FILE successfully."
