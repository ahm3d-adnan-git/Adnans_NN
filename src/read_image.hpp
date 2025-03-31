#ifndef READ_IMAGE_HPP
#define READ_IMAGE_HPP

#include <fstream>
#include <cstdint>
#include "utilities.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using Eigen::ArrayXXd;

MatrixXd getImageFromIndex(const std::vector<MatrixXd>& images, size_t imageIndex) {
    return images.at(imageIndex);  // Returns the image as a matrix
}

void writeSingleImageToFile(const std::vector<MatrixXd>& images, const std::string& outputFile, size_t imageIndex) {
    MatrixXd singleImage = getImageFromIndex(images, imageIndex);
    print_shape(singleImage);
    writeEigenMatrixToFile(singleImage, outputFile);
}

std::vector<MatrixXd> doReadImage(std::ifstream& inputFile) {
    uint32_t magic_number, num_of_images, rows, cols;

    inputFile.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    inputFile.read(reinterpret_cast<char*>(&num_of_images), sizeof(num_of_images));
    inputFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inputFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    magic_number = swapEndian(magic_number);
    num_of_images = swapEndian(num_of_images);
    rows = swapEndian(rows);
    cols = swapEndian(cols);

    std::vector<MatrixXd> images;

    for (uint32_t i = 0; i < num_of_images; ++i) {
        MatrixXd image(rows, cols);  // Store image as a 2D matrix
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint8_t pixel;
                inputFile.read(reinterpret_cast<char*>(&pixel), 1);
                image(r, c) = static_cast<double>(pixel) / 255.0; // Normalize pixel values
            }
        }
        images.push_back(image);  // Add matrix to vector
    }

    return images;
}
#endif