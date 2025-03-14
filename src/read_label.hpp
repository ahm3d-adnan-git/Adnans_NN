#ifndef READ_LABEL_HPP
#define READ_LABEL_HPP

#include <fstream>
#include <vector>
#include <cstdint>
#include "tensor.hpp"
#include "utilities.hpp"

Tensor<double> getLabelFromIndex(Tensor<double>& labels,size_t labelIndex) {
    Tensor<double> label({10});
    for (size_t r = 0; r < 10; ++r) {

        label({r}) = labels({labelIndex, r});
    }
    return label;
}

void writeSingleLabelToFile(Tensor<double>& labels,const std::string& outputFile,size_t imageIndex) {
    Tensor<double> singleLabel({10});
    // Write the single image tensor to a file
    singleLabel = getLabelFromIndex(labels,imageIndex);
    writeTensorToFile(singleLabel, outputFile);

}



Tensor<double> doReadLabel(std::ifstream& inputFile) {
   
    uint32_t magic_number,num_of_labels=0;
    inputFile.read((char*)&magic_number, sizeof(magic_number));

    inputFile.read((char*)&num_of_labels, sizeof(num_of_labels));

    magic_number = swapEndian(magic_number);
    num_of_labels = swapEndian(num_of_labels);

    if (num_of_labels == 1) {
        uint8_t label;
        inputFile.read((char*) &label, 1);
        Tensor<double> one_hot_label({10}, 0.0);
        one_hot_label({label}) = 1.0;
        return one_hot_label;
    }

    Tensor<double> labels({num_of_labels, 10}, 0.0);
    for (uint32_t i = 0; i < num_of_labels; ++i) {

        uint8_t label;
        inputFile.read((char*) &label, 1);

        labels({i, label}) = 1.0;

    }
    return labels;
}


#endif