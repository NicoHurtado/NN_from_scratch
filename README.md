# Neural Network from Scratch

This project implements a simple feedforward neural network from scratch using **NumPy**, without any deep learning frameworks like TensorFlow or PyTorch. It is trained on the **MNIST-style digit dataset** (`train.csv`) for classifying handwritten digits (0–9).

## What This Project Does

- Loads and preprocesses image data
- Implements forward propagation (with ReLU and softmax activations)
- Performs backpropagation using cross-entropy loss
- Trains a 2-layer neural network via gradient descent
- Makes predictions and visualizes test samples

---

## Model Architecture

- **Input layer:** 784 neurons (28×28 grayscale image)
- **Hidden layer:** 10 neurons (ReLU activation)
- **Output layer:** 10 neurons (Softmax activation for multiclass classification)

---

## Core Functions

| Function             | Description                                                |
|----------------------|------------------------------------------------------------|
| `init_params()`      | Initializes weights and biases randomly                    |
| `forward_prop()`     | Computes forward pass (Z, A values for each layer)         |
| `backward_prop()`    | Computes gradients via backpropagation                     |
| `update_params()`    | Applies gradient descent to update weights                 |
| `get_predictions()`  | Returns predicted class indices from output probabilities  |
| `get_accuracy()`     | Calculates model accuracy                                  |
| `gradient_descent()` | Full training loop over specified number of iterations     |
| `test_prediction()`  | Predicts and displays individual digit images              |

---

## Dataset

You must place the `train.csv` file in the project root.  
This dataset is a simplified version of MNIST with each row representing:

- `label` (first column): digit 0–9
- `pixel0` to `pixel783`: flattened pixel values (0–255)

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/NicoHurtado/nn_from_scratch.git
   cd nn_from_scratch
   ```

2. Install dependencies (using Python 3.8+):
   ```bash
   pip install numpy pandas matplotlib
   ```

3. Run the notebook:
   ```bash
   jupyter notebook NN_scratch.ipynb
   ```

---

## Sample Output

The model displays handwritten digit images with the predicted label and the ground truth after training.

---

## Notes

- This project is to help understand how neural networks work under the hood.
- No external ML libraries are used—**everything is built from scratch**.

---
