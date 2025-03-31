#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Navigate to the project directory
cd "$(dirname "$0")"

echo "Executing build.sh"
./build.sh
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Executing mnist.sh"
./mnist.sh mnist-configs/input.config 
if [ $? -ne 0 ]; then
    echo "MNIST script execution failed!"
    exit 1
fi

echo "Evaluating accuracy"
#C:/Users/mdadn/AppData/Local/Programs/Python/Python311/ logs/eval_accuracy_log.py logs/log_predictions.txt

python3 logs/eval_accuracy_log.py logs/log_predictions.txt

if [ $? -ne 0 ]; then
    echo "Accuracy evaluation failed!"
    exit 1
fi

echo "All steps completed successfully!"