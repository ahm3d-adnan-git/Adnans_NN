cmake_minimum_required(VERSION 3.27)
project(ws2024_group_51_advpt_mnist)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

add_executable(ws2024_group_51_advpt_mnist
    src/mnist_loader.hpp
    src/matvec.hpp
    src/tensor.hpp
    src/utilities.hpp
    src/read_image.hpp
    src/read_label.hpp
    src/config.hpp
    src/main.cpp
    src/neural_network.hpp
)
