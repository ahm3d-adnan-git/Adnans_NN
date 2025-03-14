#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP


#include <fstream>
#include <vector>
#include <string>
#include "utilities.hpp"
#include "tensor.hpp"
#include "read_image.hpp"
#include "read_label.hpp"


class MNISTLoader {
public:
    MNISTLoader(const std::string& image_path, const std::string& label_path)
    : image_path(image_path), label_path(label_path) {}
    void load_data(){
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
    Tensor<double> get_images(){
        return images;
    }
    Tensor<double> get_labels(){
        return labels;
    }

private:
    std::string image_path, label_path;
    Tensor<double> images;
    Tensor<double> labels;

};

#endif // MNIST_LOADER_HPP