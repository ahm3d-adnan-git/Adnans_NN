cmake_minimum_required(VERSION 3.27)
project(ws2024_group_51_advpt_mnist)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Find and link OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

include_directories(src)

add_executable(ws2024_group_51_advpt_mnist
    src/mnist_loader.hpp
    src/utilities.hpp
    src/read_image.hpp
    src/read_label.hpp
    src/config.hpp
    src/main.cpp
    src/neural_network.hpp
)
