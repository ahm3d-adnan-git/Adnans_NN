echo "This script should read a dataset label into a tensor and pretty-print it into a text file..."
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
LABEL_INDEX=$3


# Check if the input file exists
if [ ! -f "$INPUT_FILE" ]; then
    log_info "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

# Check if the image index is a valid number
if ! [[ "$LABEL_INDEX" =~ ^[0-9]+$ ]]; then
    log_info "Error: Image index must be a non-negative integer."
    exit 1
fi

# Compile the main.cpp if needed
EXECUTABLE="./read_label"
if [ ! -f "$EXECUTABLE" ]; then
    g++ -o read_label src/main.cpp -std=c++20
    if [ $? -ne 0 ]; then
        log_info "Compilation failed. Please fix errors in main.cpp."
        exit 1
    fi
fi


# Run the executable
# log_info "Running the program..."
$EXECUTABLE "$INPUT_FILE" "$OUTPUT_FILE" "$LABEL_INDEX"
if [ $? -ne 0 ]; then
    log_info "Error: Failed to process the image."
    exit 1
fi


# log_info "Label $LABEL_INDEX written to $OUTPUT_FILE successfully."
