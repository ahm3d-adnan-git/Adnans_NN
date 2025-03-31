#ifndef READ_LABEL_HPP
#define READ_LABEL_HPP

#include <fstream>
#include <cstdint>
#include "Eigen/Dense"
#include "utilities.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd getLabelFromIndex(const MatrixXd& labels, size_t labelIndex) {
    if (labelIndex >= labels.rows()) {
        throw std::out_of_range("Label index out of bounds.");
    }
    return labels.row(labelIndex).transpose();
}

void writeSingleLabelToFile(const MatrixXd& labels, const std::string& outputFile, size_t imageIndex) {
    VectorXd singleLabel = getLabelFromIndex(labels, imageIndex);
    writeEigenVectorToFile(singleLabel, outputFile);
}

MatrixXd doReadLabel(std::ifstream& inputFile) {
    uint32_t magic_number, num_of_labels = 0;
    inputFile.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    inputFile.read(reinterpret_cast<char*>(&num_of_labels), sizeof(num_of_labels));

    magic_number = swapEndian(magic_number);
    num_of_labels = swapEndian(num_of_labels);

    if (num_of_labels == 1) {
        uint8_t label;
        inputFile.read(reinterpret_cast<char*>(&label), 1);
        VectorXd one_hot_label = VectorXd::Zero(10);
        one_hot_label(label) = 1.0;
        return one_hot_label.transpose();
    }

    MatrixXd labels = MatrixXd::Zero(num_of_labels, 10);
    for (uint32_t i = 0; i < num_of_labels; ++i) {
        uint8_t label;
        inputFile.read(reinterpret_cast<char*>(&label), 1);
        labels(i, label) = 1.0;
    }

    return labels;
}

#endif