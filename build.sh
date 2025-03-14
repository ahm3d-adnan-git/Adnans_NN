#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
BUILD_DIR="cmake-build-debug" # Build directory
EXECUTABLE="ws2024_group_51_advpt_mnist" # Name of the executable as defined in CMakeLists.txt

# Step 1: Check and create the build directory if it doesn't exist
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir "$BUILD_DIR"
fi

# Step 2: Navigate to the build directory
cd "$BUILD_DIR"

# Step 3: Run CMake to configure the build system
echo "Configuring project with CMake..."
cmake ..

# Step 4: Build the project using CMake
echo "Building the project..."
cmake --build .

# Step 5: Notify the user about the build status
if [ -f "$EXECUTABLE" ]; then
    echo "Build successful! Executable is located at $BUILD_DIR/$EXECUTABLE"
else
    echo "Build completed, but executable not found. Check your CMakeLists.txt for errors."
fi