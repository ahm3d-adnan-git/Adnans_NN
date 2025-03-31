#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <iostream>
#include <string>
#include <fstream>
#include "Eigen/Core"

using namespace Eigen; 

void print_shape(const MatrixXd& mat) {
    std::cout << "Shape: (" << mat.rows() << ", " << mat.cols() << ")" << std::endl;
}

void print_shape(const VectorXd& vec) {
    std::cout << "Shape: (" << vec.size() << ")" << std::endl;
}

unsigned int swapEndian(unsigned int num) {
    return ((num >> 24) & 0xFF) |      // Move byte 3 to byte 0
           ((num << 8) & 0xFF0000) |  // Move byte 1 to byte 2
           ((num >> 8) & 0xFF00) |    // Move byte 2 to byte 1
           ((num << 24) & 0xFF000000);// Move byte 0 to byte 3
}

bool compareWithExpectedFile(const char*& generatedFileName,const char*& expectedFileName) {
    std::ifstream expectedOutputFile(expectedFileName, std::ios::binary);
    std::ifstream generatedOutputFile(generatedFileName,std::ios::binary);
    if (!expectedOutputFile) {
        std::cerr << "Error opening file: " << expectedFileName << std::endl;
        return false;
    }
    if (!generatedOutputFile) {
        std::cerr << "Error opening file: " << generatedFileName << std::endl;
        return false;
    }


    std::string genLine, expectedLine;

    size_t lineNumber = 0;
    bool allMatch = true;
    while (std::getline(generatedOutputFile, genLine) && std::getline(expectedOutputFile, expectedLine)) {
        lineNumber++;
        if (genLine != expectedLine) {
            std::cerr << "Mismatch at line " << lineNumber << ":\n";
            std::cerr << "Generated: " << genLine << "\n";
            std::cerr << "Reference: " << expectedLine << "\n";
            allMatch = false;
            break;
        }
    }
    expectedOutputFile.close();
    generatedOutputFile.close();
    return allMatch;

}
// Function to write Eigen Matrix to a file
template<typename Derived>
void writeEigenMatrixToFile(const Eigen::MatrixBase<Derived>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + filename);
    }

    file << 2 << "\n";

    // Write shape (rows and cols)
    file << matrix.rows() << "\n";
    file << matrix.cols() << "\n";

    // Write data in row-major order
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j) << "\n";
        }
    }


    file.close();
}

template<typename Derived>
void writeEigenVectorToFile(const Eigen::MatrixBase<Derived>& vector, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + filename);
    }

    // Ensure input is a vector (either row or column)
    if (vector.rows() != 1 && vector.cols() != 1) {
        throw std::invalid_argument("Error: The input is not a vector.");
    }

    // Get vector size
    int vectorSize = vector.size();

    // Write rank (1 because it's a vector)
    file << "1\n";

    // Write size of vector
    file << vectorSize << "\n";

    // Set precision for uniform formatting
    file << std::fixed << std::setprecision(6);

    // Write vector elements
    for (int i = 0; i < vectorSize; i++) {
        file << vector(i) << "\n";
    }

    file.close();
}

#endif // UTILITIES_HPP

//Loader for the MNIST dataset

#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <fstream>
#include <string>
#include "Eigen/Dense"
#include "read_image.hpp"
#include "read_label.hpp"

using Eigen::MatrixXd;

class MNISTLoader {
public:
    MNISTLoader(const std::string& image_path, const std::string& label_path)
        : image_path(image_path), label_path(label_path) {}

    void load_data() {
        std::ifstream imageFile(image_path, std::ios::binary);
        if (!imageFile.is_open()) {
            throw std::runtime_error("Error opening image file: " + image_path);
        }
        images = doReadImage(imageFile);

        std::ifstream labelFile(label_path, std::ios::binary);
        if (!labelFile.is_open()) {
            throw std::runtime_error("Error opening label file: " + label_path);
        }
        labels = doReadLabel(labelFile);

        labelFile.close();
        imageFile.close();
    }

    std::vector<MatrixXd> get_images() {
        return images;
    }

    MatrixXd get_labels() {
        return labels;
    }

private:
    std::string image_path, label_path;
    std::vector<MatrixXd> images;  // Now storing images as 2D matrices
    MatrixXd labels;
};

#endif // MNIST_LOADER_HPP