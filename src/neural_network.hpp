#pragma once
#include "Eigen/Dense"
#include <string>
#include <omp.h>

#define EIGEN_DONT_PARALLELIZE  // Disable Eigen's internal parallelization

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size,
                  double learning_rate, int batch_size, int epochs,
                  const std::string& rel_path_log_file)
        : learning_rate(learning_rate), batch_size(batch_size), epochs(epochs),
          rel_path_log_file(rel_path_log_file) {

        // Xavier Initialization for Weights
        double stddev_W1 = sqrt(2.0 / (input_size + hidden_size));
        double stddev_W2 = sqrt(2.0 / (hidden_size + output_size));

        W1 = MatrixXd::Random(hidden_size, input_size) * stddev_W1;
        W2 = MatrixXd::Random(output_size, hidden_size) * stddev_W2;

        b1 = VectorXd::Zero(hidden_size);
        b2 = VectorXd::Zero(output_size);
    }
    void forward(const VectorXd& X, VectorXd& A1, VectorXd& A2) {
        VectorXd Z1 = W1 * X + b1;
        A1 = Z1.array().max(0);  // ReLU activation
        
        // Hidden to output layer
        VectorXd Z2 = W2 * A1 + b2;
        A2 = softmax(Z2);
        
    }
    void train(const MatrixXd& X, const MatrixXd& Y) {
        std::cout << "Training started..." << std::endl;
        double epoch_loss = 0.0;
        for (size_t i = 0; i < epochs; i++) {
            epoch_loss = 0.0;
            int num_samples = X.rows();  // Number of training samples

            #pragma omp parallel for reduction(+:epoch_loss)
            for (int j = 0; j < num_samples; j += batch_size) {
                size_t end = std::min(j + batch_size, num_samples);
                double local_loss = 0.0;

                for (size_t k = j; k < end; ++k) {
                    VectorXd X_sample = X.row(k).transpose();  // Convert each sample to VectorXd
                    VectorXd Y_sample = Y.row(k).reshaped(Y.cols(), 1);

                    // Forward pass
                    VectorXd A1, A2;
                    forward(X_sample, A1, A2);
                    // Compute loss (optional tracking for debugging)
                    if (Y_sample.size() != A2.size()) {
                        Y_sample = Y_sample.replicate(A2.size() / Y_sample.size(), 1);
                    }
                    double loss = cross_entropy_loss(A2, Y_sample);
                    
                    epoch_loss += loss;

                    // Compute gradients
                    MatrixXd dW1 = MatrixXd::Zero(W1.rows(), W1.cols());
                    MatrixXd dW2 = MatrixXd::Zero(W2.rows(), W2.cols());
                    VectorXd db1 = VectorXd::Zero(b1.size());
                    VectorXd db2 = VectorXd::Zero(b2.size());
                    

                    backward(X_sample, Y_sample, A1, A2, W1, W2, dW1, db1, dW2, db2);
                    local_loss += loss;
                    // Update parameters
                    update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate);
                }
                #pragma omp atomic
                epoch_loss += local_loss;
            }
            std::cout << "Epoch " << i << " Loss: " << epoch_loss << std::endl;
        }
    }
    void test(const MatrixXd& X, const MatrixXd& Y) {
        size_t num_samples = X.rows();
        std::ofstream log_file(rel_path_log_file);
        if (!log_file) {
            std::cerr << "Error: Could not open log file!" << std::endl;
            return;
        }

        size_t correct_predictions = 0;
        size_t batch_index = 0;

        for (size_t i = 0; i < num_samples; i += batch_size) {
            size_t batch_end = std::min(i + batch_size, num_samples);
            log_file << "Current batch: " << batch_index++ << std::endl;
            // Process each sample individually using VectorXd
            #pragma omp parallel for
            for (size_t j = i; j < batch_end; ++j) {
                VectorXd X_sample = X.row(j).transpose();  // Convert input row to VectorXd
                VectorXd Y_sample = Y.row(j).transpose();  // Convert label row to VectorXd
                VectorXd A1, A2;
                forward(X_sample, A1, A2);  // Forward pass for single sample
                
                // Extract predicted label (index of max value in A2)
                int predicted_label = 0;
                A2.maxCoeff(&predicted_label);

                // Extract true label (index of max value in Y_sample)
                int true_label = 0;
                Y_sample.maxCoeff(&true_label);

                // Compare predictions
                if (predicted_label == true_label) {
                    correct_predictions++;
                }
                #pragma omp critical
                {
                    std::cout << "- image " << j << ": Prediction=" << predicted_label
                            << ". Label=" << true_label << "\n";
                    log_file << "- image " << j << ": Prediction=" << predicted_label
                            << ". Label=" << true_label << "\n";
                }
            }
        }

        double accuracy = static_cast<double>(correct_predictions) / num_samples;
        std::cout << "Accuracy %: " << accuracy * 100 << "%" << std::endl;
        log_file.close();
    }

    MatrixXd get_W1() { return W1; }
    MatrixXd get_W2() { return W2; }
    VectorXd get_b1() { return b1; }
    VectorXd get_b2() { return b2; }
private:
    double learning_rate;
    MatrixXd W1, W2;
    VectorXd b1, b2;
    int batch_size, epochs;
    std::string rel_path_log_file;

    VectorXd softmax(const VectorXd& logits) {
        VectorXd exp_values = logits.array().exp();
        return exp_values / exp_values.sum();
    }

    double cross_entropy_loss(const VectorXd& Y_pred, const VectorXd& Y_true) {
        // Clip values to avoid log(0) issues
        VectorXd  clipped_Y_pred = Y_pred.array().max(1e-10);
        // Compute cross-entropy loss
        VectorXd  loss = (-Y_true.array() * clipped_Y_pred.array().log()).matrix();

        // Return the average loss over all samples
        return loss.sum() / Y_pred.cols();
    }

    void backward(const VectorXd& X, const VectorXd& Y,
              const VectorXd& A1, const VectorXd& A2,
              const MatrixXd& W1, const MatrixXd& W2,
              MatrixXd& dW1, VectorXd& db1,
              MatrixXd& dW2, VectorXd& db2) {
        VectorXd dA2 = A2 - Y;  // Softmax derivative
        dW2 = dA2 * A1.transpose();
        db2 = dA2;
        VectorXd dA1 = W2.transpose() * dA2;
        VectorXd dF = (A1.array() > 0).cast<double>();  // ReLU derivative
        
        dA1 = dA1.array() * dF.array();

        dW1 = dA1 * X.transpose();

        db1 = dA1;

    }

    void update_parameters(MatrixXd& W1, VectorXd& b1,
                       MatrixXd& W2, VectorXd& b2,
                       const MatrixXd& dW1, const VectorXd& db1,
                       const MatrixXd& dW2, const VectorXd& db2,
                       double learning_rate) {
        W1 -= learning_rate * dW1;
        b1 -= learning_rate * db1;
        W2 -= learning_rate * dW2;
        b2 -= learning_rate * db2;
    }
};