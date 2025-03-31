# 🧠 MNIST Handwriting Recognition - C++ Neural Network

This project is a complete implementation of a fully-connected neural network (FCNN) in **C++** for recognizing handwritten digits using the **MNIST dataset**. Built from scratch without high-level ML libraries, it offers deep insight into how neural networks operate at the low level.

---

## 📌 Features

- Loads and parses the binary MNIST dataset (images + labels)
- Normalizes grayscale pixel data to [0, 1]
- One-hot encodes labels (0–9)
- Implements:
  - Forward propagation
  - ReLU & Softmax activations
  - Cross-entropy loss
  - Backpropagation
  - Stochastic Gradient Descent (SGD)
- Configurable:
  - Hidden layer size
  - Learning rate
  - Epochs
  - Batch size
- Evaluation and accuracy reporting
- Prediction logging in required format for validation

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `main.cpp` | Entry point: parses config, loads data, trains and tests model |
| `config.hpp` | Parses config file for hyperparameters and file paths |
| `read_image.hpp` | Reads MNIST image binary files and normalizes to matrices |
| `read_label.hpp` | Reads MNIST label files and one-hot encodes them |
| `mnist_loader.hpp` | Wraps image + label loading and provides dataset access |
| `neural_network.hpp` | Neural network logic: forward, backward, training, testing |
| `utilities.hpp` | Helper functions (file writing, shape printing, endian conversion) |

---

## ⚙️ How to Run

### 1. Run the run.sh file
You should edit the Line which points to your python and then run the run.sh bash script.

