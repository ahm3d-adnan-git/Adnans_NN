#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP


#include "tensor.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size, double learning_rate, bool is_single_sample,int batch_size,int epochs,std::string rel_path_log_file){
        // Constructor: Initialize weight tensors using Xavier initialization
        // - Initialize weights and biases for each layer
        // - Use Xavier initialization for weights
        // - Initialize biases to zero
        W1 = Tensor<double>({hidden_size, input_size});
        W1 = W1.he_init();
        b1 = Tensor<double>({hidden_size, 1}, 0.0);
        W2 = Tensor<double>({output_size, hidden_size});
        W2 = W2.he_init();
        b2 = Tensor<double>({output_size, 1}, 0.0);
        this->learning_rate = learning_rate;
        this->batch_size = batch_size;
        this->epochs = epochs;
        this->rel_path_log_file = rel_path_log_file;
    }
    void forward(const Tensor<double>& X,
             Tensor<double>& W1, const Tensor<double>& b1, 
             const Tensor<double>& W2, const Tensor<double>& b2, 
             Tensor<double>& A1, Tensor<double>& A2)
    {
        // Forward function: Compute activations through layers
        // - Apply weights and biases
        // - Use sigmoid and softmax activation functions
        // std::cerr << "before forward calculation" << std::endl;
        // std::cerr << "X shape" << std::endl;
        // print_shape(X.shape());
        // std::cerr << "W1 shape" << std::endl;
        // print_shape(W1.shape());
        // std::cerr << "b1 shape" << std::endl;
        // print_shape(b1.shape());
        // std::cerr << "W2 shape" << std::endl;
        // print_shape(W2.shape());
        // std::cerr << "b2 shape" << std::endl;
        // print_shape(b2.shape());
        // std::cerr << "forward before W1.dot(X.T()) + b1" << std::endl;
        size_t X_size = X.shape()[0];
        // std::cerr << "batch_size" << std::endl;
        // std::cerr << batch_size << std::endl;
        // std::cerr << "b1.shape()[0]" << std::endl;
        // std::cerr << b1.shape()[0] << std::endl;
        // repeat(1, batch_size) for b1 to broadcast bias over batch dimension
        // change b1 size if it is not the same as batch size
        if (b1.shape()[0] != batch_size) {
            A1 = W1.dot(X.T()) + b1.repeat(1, X_size);
        }
        else {
            A1 = W1.dot(X.T()) + b1;
        }
        // std::cerr << "forward after W1.dot(X) + b1" << std::endl;
        // std::cerr << "A1 shape" << std::endl;
        // print_shape(A1.shape());

        A1 = A1.relu();
        
        // std::cerr << "forward after A1.relu()" << std::endl;
        // std::cerr << "A1 shape" << std::endl;
        // print_shape(A1.shape());
        // repeat(1, batch_size) for b2 to broadcast bias over batch dimension
        if(b2.shape()[0] != X_size){
            A2 = W2.dot(A1) + b2.repeat(1, X_size);
        }
        else{
            A2 = W2.dot(A1) + b2;
        }

        // std::cerr << "forward after W2.dot(A1) + b2" << std::endl;
        // std::cerr << "A2 shape" << std::endl;
        // print_shape(A2.shape());

        A2 = A2.softmax();
        
        // std::cerr << "forward after softmax" << std::endl;
        // std::cerr << "A2 shape" << std::endl;
        // print_shape(A2.shape());
        // std::cerr << "forward finished" << std::endl;
    }

    // Train the network using given data, labels, epochs, and batch size
    void train(const Tensor<double>& X, const Tensor<double>& Y)
    {
        // Forward function: Compute activations through layers
        // - Apply weights and biases
        // - Use sigmoid and softmax activation functions
        // Compute loss
        // Backward function: Compute gradients and update weights
        // - Use gradient descent to update weights
        for (size_t i = 0; i < epochs; i++)
        {
            Tensor<double> A1, A2;
            // std::cout << "Training the network..." << std::endl;
            // std::cout << "X.shape(): " << std::endl;
            // print_shape(X.shape());
            // std::cout << "X.rank(): " << X.rank() << std::endl;
            // std::cout << "X.shape()[0]: " << X.shape()[0] << std::endl;
            int num_samples = X.shape()[0]; // If single image, num_samples will be 1
            // std::cout << "Number of samples: (X.rank() == 2) ? 1 : X.shape()[0] : " << num_samples << std::endl;
            for (int j = 0; j < num_samples; j += batch_size)
            {
                // std::cout << "start of for loop" << std::endl;
                // std::cout << "j: " << j << std::endl;
                // std::cout << "min(j + batch_size, num_samples) " << std::endl;
                size_t end = std::min((j + batch_size), num_samples); // Ensure not exceeding dataset size
                
                // std::cout << "batch_end: " << end << std::endl;
                Tensor<double> X_batch, Y_batch;
                X_batch = X.slice(j, end);
                Y_batch = Y.slice(j, end);
                forward(X_batch, W1, b1, W2, b2, A1, A2);
                if (Y_batch.shape() != A2.shape()) {
                    Y_batch = Y_batch.reshape(A2.shape()); // Reshaping Y_batch to match A2... Make sure labels match A2 dimensions
                }
                // Compute loss (optional for monitoring)
                double loss = cross_entropy_loss(A2, Y_batch);
                // std::cerr << "Epoch " << i << " - Loss: " << loss << std::endl;
                // std::cerr << "W1 norm: " << W1.sum() << ", W2 norm: " << W2.sum() << std::endl;

                // Backward pass
                Tensor<double> dW1, db1, dW2, db2;
                backward(X_batch, Y_batch, A1, A2, W1, W2, dW1, db1, dW2, db2);
                // 🔥 Gradient Clipping to prevent exploding gradients
                double clip_value = 2.0;
                dW1 = dW1.clip(-clip_value, clip_value);
                db1 = db1.clip(-clip_value, clip_value);
                dW2 = dW2.clip(-clip_value, clip_value);
                db2 = db2.clip(-clip_value, clip_value);
                // std::cerr << "Gradient norms - dW1: " << dW1.sum() << ", dW2: " << dW2.sum() << std::endl;
                // Update weights and biases
                update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate);
            }

        }
        
    }
    void test(const Tensor<double>& X, const Tensor<double>& Y){
        // std::cout << "Testing the network..." << std::endl;
        size_t num_samples = X.shape()[0];
        // std::cout << "X.shape(): " << std::endl;
        // print_shape(X.shape());
        // std::cout << "X.rank(): " << X.rank() << std::endl;
        // std::cout << "X.shape()[0]: " << X.shape()[0] << std::endl;
        
        // std::cout << "Number of samples: (X.rank() == 2) ? 1 : X.shape()[0] : " << num_samples << std::endl;

        std::string log_filename = rel_path_log_file;
        // std::cout << "Log file: " << log_filename << std::endl;
        std::ofstream log_file(log_filename);
        if (!log_file) {
            std::cerr << "Error: Could not open log file!" << std::endl;
            return;
        }
        // X.rank() == 2 means single image, rank() == 3 means batch
        // If single image, num_samples = 1, else use batch size
        // std::cout << "X.rank() == 2 means single image, rank() == 3 means batch" << std::endl;
        // std::cout << "If single image, num_samples = 1, else use batch size" << std::endl;
        // std::cout << "num_samples: " << num_samples << std::endl;
        // std::cout << "batch_size: " << batch_size << std::endl;
        size_t correct_predictions = 0;
        size_t batch_index = 0;
        for (size_t i = 0; i < num_samples; i += batch_size) {
            // std::cout << "start of for loop" << std::endl;
            // std::cout << "i: " << i << std::endl;
            // std::cout << "min(i + batch_size, num_samples) " << std::endl;
            size_t batch_end = std::min(i + batch_size, num_samples);
            // std::cout << "batch_end: " << batch_end << std::endl;
            log_file << "Current batch: " << batch_index++ << std::endl;
            std::cout << "Current batch: " << batch_index << std::endl;
                for (size_t j = i; j < batch_end; ++j) {
                    // Extract single sample
                    Tensor<double> X_sample = X.slice(j, j + 1);
                    Tensor<double> Y_sample = Y.slice(j, j + 1);

                    // Perform forward pass
                    Tensor<double> A1, A2;
                    forward(X_sample, W1, b1, W2, b2 ,A1, A2);

                    if (A2.shape() != Y_sample.shape()) {
                        Y_sample = Y_sample.reshape(A2.shape());
                    }
                    // Get predicted label
                    Tensor<double> predicted_labels = A2.argmax(0);
                    Tensor<double> true_labels = Y_sample.argmax(0);

                    // Check if prediction matches true label
                    for (size_t k = 0; k < predicted_labels.shape()[0]; ++k) {
                        if (predicted_labels({k}) == true_labels({k})) {
                            correct_predictions++;
                            // Write to log file
                            std::cout << "- image " << j <<": Prediction=" << predicted_labels({k}) << ". Label=" << true_labels({k}) << "\n";
                            log_file << "- image " << j <<": Prediction=" << predicted_labels({k}) << ". Label=" << true_labels({k}) << "\n";
                        }
                    }

                }
            }

            // Compute accuracy
            // std::cout << "Correct predictions: " << correct_predictions << "\n";
            // std::cout << "Total samples: " << num_samples << "\n";
            double accuracy = static_cast<double>(correct_predictions) / num_samples;
            std::cout << "Accuracy % :" << accuracy * 100 << "%" << std::endl;
            log_file.close();
            
    }

    Tensor<double> get_W1(){return W1;}
    Tensor<double> get_W2(){return W2;}
    Tensor<double> get_b1(){return b1;}
    Tensor<double> get_b2(){return b2;}
private:
    double learning_rate;
    Tensor<double> W1;
    Tensor<double> b1;
    Tensor<double> W2;
    Tensor<double> b2;
    int batch_size;
    int epochs;
    std::string rel_path_log_file;
    
    // Compute loss (cross-entropy loss for classification)
    double cross_entropy_loss(const Tensor<double>& Y_pred, const Tensor<double>& Y_true) {
        // Small epsilon value to prevent log(0)
        const double epsilon = 1e-10;

        // Ensure Y_pred is numerically stable by clipping values
        Tensor<double> clipped_Y_pred = Y_pred.clip(epsilon, 1.0); // Clip values to avoid log(0)

        // Compute element-wise cross-entropy loss
        Tensor<double> loss = Y_true * clipped_Y_pred.log(); // Use natural log (ln)

        // Return the negative sum of the mean loss over all samples
        return -loss.sum() / Y_pred.shape()[0]; // Normalize over batch size
    }

    
    // Backward pass: Compute gradients and update weights
    void backward(const Tensor<double>& X, const Tensor<double>& Y, 
              const Tensor<double>& A1, const Tensor<double>& A2, 
              const Tensor<double>& W1, const Tensor<double>& W2, 
              Tensor<double>& dW1, Tensor<double>& db1, 
              Tensor<double>& dW2, Tensor<double>& db2)
    {
        // Backpropagation (Backward pass)
        // - Compute gradient of loss with respect to weights and biases
        // - Update weights using gradient descent
        
        // std::cout << "backprop started" << std::endl;
        // std::cout << "before dA2 = A2 - Y;" << std::endl;
        // std::cout << "A2 shape" << std::endl;
        // print_shape(A2.shape());
        // std::cout << "Y shape" << std::endl;
        // print_shape(Y.shape());

        Tensor<double> dA2 = A2 - Y;
        // std::cout << "dW2 = dA2.dot(A1.T());" << std::endl;
        // std::cout << "dA2 shape" << std::endl;
        // print_shape(dA2.shape());
        dW2 = dA2.dot(A1.T());
        // std::cout << "dW2 shape" << std::endl;
        // std::cout << "db2 = dA2.sum(1);" << std::endl;
        db2 = dA2.sum(1);
        // std::cout << "db2 shape after reshape" << std::endl;
        // print_shape(db2.shape());
        
        // std::cout << "dA1 = W2.T().dot(dA2);" << std::endl;
        Tensor<double> dA1 = W2.T().dot(dA2);
        // std::cout << "dA1 shape" << std::endl;
        // print_shape(dA1.shape());
        // std::cout << "dF = A1.relu_derivative();" << std::endl;
        Tensor<double> dF = A1.relu_derivative();
        // std::cout << "dF shape" << std::endl;
        // print_shape(dF.shape());
        
        // std::cout << " dA1 = dA1 * dF;" << std::endl;
        dA1 = dA1 * dF;
        // std::cout << "dA1 shape" << std::endl;
        // print_shape(dA1.shape());
        // std::cout << "dW1 = dA1.dot(X);" << std::endl;
        dW1 = dA1.dot(X); // Correction: (500,1) * (1,784) = (500, 784)
        // std::cout << "dW1 shape" << std::endl;
        // print_shape(dW1.shape());
        // std::cout << "db1 = dA1.sum(1);" << std::endl;
        db1 = dA1.sum(1); // Sum over batch dimension
        // std::cout << "db1 shape after dA1.sum(1) and reshape" << std::endl;
        // print_shape(db1.shape());

        // std::cout << "backprop finished" << std::endl;
    }
 
    void update_parameters(Tensor<double>& W1, Tensor<double>& b1, 
                       Tensor<double>& W2, Tensor<double>& b2,
                       const Tensor<double>& dW1, const Tensor<double>& db1,
                       const Tensor<double>& dW2, const Tensor<double>& db2, 
                       double learning_rate)
    {
        // Update weights and biases using SGD
        // std::cout << "update_parameters started" << std::endl;
       
        // std::cout << "W1 - (dW1 * learning_rate);" << std::endl;
        // std::cout << "dW1 shape" << std::endl;
        // print_shape(dW1.shape());
        Tensor<double> W1_update_term = dW1 * learning_rate;
        // std::cout << "W1_update_term shape" << std::endl;
        // print_shape(W1_update_term.shape());

        // std::cout << "b1 - (db1 * learning_rate);" << std::endl;
        // std::cout << "db1 shape" << std::endl;
        // print_shape(db1.shape());
        Tensor<double> b1_update_term = db1 * learning_rate;
        // std::cout << "b1_update_term shape" << std::endl;
        // print_shape(b1_update_term.shape());

        // std::cout << "W2 - (dW2 * learning_rate);" << std::endl;
        // std::cout << "dW2 shape" << std::endl;
        // print_shape(dW2.shape());
        Tensor<double> W2_update_term = dW2 * learning_rate;
        // std::cout << "W2_update_term shape" << std::endl;
        // print_shape(W2_update_term.shape());

        // std::cout << "b2 - (db2 * learning_rate);" << std::endl;
        // std::cout << "db2 shape" << std::endl;
        // print_shape(db2.shape());
        Tensor<double> b2_update_term = db2 * learning_rate;
        // std::cout << "b2_update_term shape" << std::endl;
        // print_shape(b2_update_term.shape());
        
        // update W1, b1, W2, b2
        // std::cout << "W1 - W1_update_term;" << std::endl;
        // std::cout << "W1 shape" << std::endl;
        // print_shape(W1.shape());
        W1 = W1 - W1_update_term;
        b1 = b1 - b1_update_term;
        W2 = W2 - W2_update_term;
        b2 = b2 - b2_update_term;
        // std::cout << "update_parameters finished" << std::endl;
    }
};

#endif