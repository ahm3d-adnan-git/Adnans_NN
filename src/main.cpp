#include <vector>
#include <fstream>
#include <cstdint>
#include <iomanip>
#include <filesystem>
#include <random>
#include <cmath>
#include "read_image.hpp"
#include "read_label.hpp"
#include "mnist_loader.hpp"
#include "config.hpp"
#include "tensor.hpp"
#include "neural_network.hpp"


int main( int argc, char* argv[] ) {
    const char* command = argv[0];

    const char* inputFileName = argv[1];
    

    std::ifstream inputFile(inputFileName, std::ios::binary);
    if (!inputFile.is_open())
    {
        std::cerr << "Could not open input file." << std::endl;
        std::exit(1);
    }
    if(argc >2){
        const char* outputFileName = argv[2];
        size_t imageIndex = std::stoul(argv[3]);
        if (command == std::string("./read_label")){
            Tensor<double> labels = doReadLabel(inputFile);
            writeSingleLabelToFile(labels,outputFileName,imageIndex);
        }
        else if (command == std::string("./read_image")){
            Tensor<double> images = doReadImage(inputFile);
            writeSingleImageToFile(images, outputFileName, imageIndex);
        }
    }
    else if(command == std::string("./read_config")) {
        Config cfg(inputFileName);
        MNISTLoader mnist_loader(cfg.rel_path_train_images, cfg.rel_path_train_labels);
        mnist_loader.load_data();
        Tensor<double> images = mnist_loader.get_images();
        Tensor<double> labels = mnist_loader.get_labels();
        bool is_single_sample = false;
        if (images.rank() == 2) {
            is_single_sample = true;
        }
        NeuralNetwork nn(static_cast<size_t>(784), static_cast<size_t>(cfg.hidden_size), static_cast<size_t>(10), cfg.learning_rate, is_single_sample,cfg.batch_size,50,cfg.rel_path_log_file);
        
        // std::cout << "images rank: " << images.rank() << std::endl;
        // std::cout << "images shape: " << std::endl;
        // print_shape(images.shape());
        // std::cout << "labels rank: " << labels.rank() << std::endl;
        // std::cout << "labels shape: " << std::endl;
        // print_shape(labels.shape());
        if (images.rank() == 3 && images.shape()[1] == 28 && images.shape()[2] == 28) {
            images = images.reshape({images.shape()[0], 784});  // Convert (num_samples, 28, 28) → (num_samples, 784)
        } else if (images.rank() < 3) {
            images = images.reshape({1, 784});  // Ensure single image is reshaped properly
        }
        
        if (labels.rank() == 1) {
            if (labels.shape()[0] == 10) {
                labels = labels.reshape({1, 10});  // Ensure shape consistency
            } else {
                labels = labels.reshape({labels.shape()[0], 1});  // Convert (batch_size,) → (batch_size, 1) for proper indexing
            }
        }
        // std::cout << "after reshape: " << std::endl;
        // std::cout << "images rank: " << images.rank() << std::endl;
        // std::cout << "images shape: " << std::endl;
        // print_shape(images.shape());
        // std::cout << "labels rank: " << labels.rank() << std::endl;
        // std::cout << "labels shape: " << std::endl;
        // print_shape(labels.shape());

        nn.train(images, labels);
        nn.test(images, labels);
       
       
        
    }
    else {
        std::cerr << "Invalid mode." << std::endl;
        return 1;
    }

    inputFile.close();
    return 0;
}