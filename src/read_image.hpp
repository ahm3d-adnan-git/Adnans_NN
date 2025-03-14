#ifndef READ_IMAGE_HPP
#define READ_IMAGE_HPP

#include <fstream>
#include <vector>
#include <cstdint>
#include "tensor.hpp"
#include "utilities.hpp"

Tensor<double> getImageFromIndex(Tensor<double>& images,size_t imageIndex) {

    Tensor<double> image({images.shape()[1], images.shape()[2]});
    for (size_t r = 0; r < images.shape()[1]; ++r) {
        for (size_t c = 0; c < images.shape()[2]; ++c) {
            image({r, c}) = images({imageIndex, r, c});
        }
    }
    return image;
}

void writeSingleImageToFile(Tensor<double>& images,const std::string& outputFile,size_t imageIndex) {
    Tensor<double> singleImage({images.shape()[1], images.shape()[2]});
    // Write the single image tensor to a file
    singleImage = getImageFromIndex(images,imageIndex);
    writeTensorToFile(singleImage, outputFile);

}


Tensor<double> doReadImage(std::ifstream& inputFile) {
    uint32_t magic_number,num_of_images,cols,rows=0;
    inputFile.read((char*)&magic_number, sizeof(magic_number));

    inputFile.read((char*)&num_of_images, sizeof(num_of_images));

    inputFile.read((char*)&cols, sizeof(cols));

    inputFile.read((char*)&rows, sizeof(rows));

    magic_number = swapEndian(magic_number);
    num_of_images = swapEndian(num_of_images);
    cols = swapEndian(cols);
    rows = swapEndian(rows);
    
    if (num_of_images == 1) {
        Tensor<double> image({rows, cols});
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint8_t pixel;
                inputFile.read((char*) &pixel, 1);
                image({r, c}) = static_cast<double>(pixel) / 255.0; // Normalize pixel value
            }
        }
        return image; // Return as a single 2D image
    }

    Tensor<double> images({num_of_images, rows, cols});
    for (uint32_t i = 0; i < num_of_images; ++i) {
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < cols; ++c) {
                uint8_t pixel;
                inputFile.read((char*) &pixel, 1);
                images({i, r, c}) = static_cast<double>(pixel) / 255.0; // Normalize pixel value

            }
        }
    }
    return images;
}

#endif