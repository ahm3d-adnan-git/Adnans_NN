#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include "read_image.hpp"
#include "read_label.hpp"
#include "mnist_loader.hpp"
#include "config.hpp"
#include "neural_network.hpp"

#include <omp.h>
#define EIGEN_DONT_PARALLELIZE  // Disable Eigen's internal parallelization

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::setNbThreads;

int main(int argc, char* argv[]) {


    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file] [image_index]" << std::endl;
        return 1;
    }

    const std::string command = argv[0];
    const std::string inputFileName = argv[1];

    std::ifstream inputFile(inputFileName, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Could not open input file: " << inputFileName << std::endl;
        return 1;
    }

    if (argc > 3) {
        const std::string outputFileName = argv[2];
        size_t imageIndex = std::stoul(argv[3]);  // Safely parse image index

        if (command == "./read_label") {
            MatrixXd labels = doReadLabel(inputFile);
            writeSingleLabelToFile(labels, outputFileName, imageIndex);
        } 
        else if (command == "./read_image") {
            std::vector<MatrixXd> images = doReadImage(inputFile);

            if (imageIndex >= images.size()) {
                std::cerr << "Error: Image index out of bounds. Available images: " << images.size() << std::endl;
                return 1;
            }

            print_shape(images[imageIndex]); // Print shape of selected image
            writeSingleImageToFile(images, outputFileName, imageIndex);
        }
    } 
    else if (command == "./read_config") {
        Config cfg(inputFileName);
        MNISTLoader mnist_loader(cfg.rel_path_train_images, cfg.rel_path_train_labels);
        mnist_loader.load_data();

        std::vector<MatrixXd> images = mnist_loader.get_images(); // Updated to return a vector
        MatrixXd labels = mnist_loader.get_labels();

        NeuralNetwork nn(784, cfg.hidden_size, 10, cfg.learning_rate, cfg.batch_size, cfg.num_epochs, cfg.rel_path_log_file);

        // Convert image vector to single training matrix
        MatrixXd imageMatrix(images.size(), images[0].size());
        for (size_t i = 0; i < images.size(); ++i) {
            imageMatrix.row(i) = Eigen::Map<const Eigen::RowVectorXd>(images[i].data(), images[i].size());
        }
        
        nn.train(imageMatrix, labels);

        MNISTLoader mnist_test_loader(cfg.rel_path_test_images, cfg.rel_path_test_labels);
        
        mnist_test_loader.load_data();

        std::vector<MatrixXd> test_images = mnist_test_loader.get_images();
        MatrixXd test_labels = mnist_test_loader.get_labels();
        

        // Convert image vector to single training matrix
        MatrixXd imageTestMatrix(test_images.size(), test_images[0].size());
        for (size_t i = 0; i < test_images.size(); ++i) {
            imageTestMatrix.row(i) = Eigen::Map<const Eigen::RowVectorXd>(test_images[i].data(), test_images[i].size());
        }


        nn.test(imageTestMatrix, test_labels);
    } 
    else {
        std::cerr << "Invalid mode." << std::endl;
        return 1;
    }

    inputFile.close();
    return 0;
}
