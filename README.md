# XOR Neural Network Project

This project implements a neural network from scratch to solve the XOR problem. The XOR problem is a classic benchmark for evaluating the capabilities of a neural network, as it is not linearly separable. The implementation uses fundamental concepts of forward and backward propagation without relying on any deep learning frameworks.

## Features

- Fully connected layers implemented via a custom `Dense` class.
- Activation functions (e.g., Tanh) for non-linear transformations.
- Training using backpropagation and Mean Squared Error (MSE) loss.
- Visualization of the trained network's predictions in 3D.

## Project Structure

```
project/
├── xor.py              # Main script for defining and training the neural network
├── model.py            # Functions for forward pass and training
├── activations.py      # Tanh activation function
├── losses.py           # Implementation of MSE loss and its derivative
├── layer.py            # Base Layer class
├── activation.py       # Base Activation class inheriting from Layer
├── dense.py            # Fully connected Dense layer implementation
```

## Installation

This project requires Python and the following libraries:

- `numpy`
- `matplotlib`

Install the dependencies using:

```bash
pip install numpy matplotlib
```

## How It Works

### 1. Network Structure

The network is defined as:

- Input layer: 2 neurons
- Hidden layer: 3 neurons with Tanh activation
- Output layer: 1 neuron with Tanh activation

### 2. Training

The network is trained using the Mean Squared Error (MSE) loss and backpropagation. The training loop:

- Computes the forward pass to predict outputs.
- Calculates the loss and propagates gradients backward.
- Updates weights and biases using gradient descent.

### 3. Visualization

After training, the network's predictions for the XOR problem are visualized in 3D using Matplotlib.

## Example Usage

Run the `xor.py` script to train the network and visualize the results:

```bash
python xor.py
```

## Key Components

### `xor.py`

- Defines the XOR dataset.
- Builds the network architecture using `Dense` layers and Tanh activation.
- Trains the network and visualizes the results.

### `model.py`

- **`predict`**: Performs a forward pass through the network.
- **`train`**: Trains the network using MSE loss and backpropagation.

### `activations.py`

- Implements the Tanh activation function and its derivative.

### `losses.py`

- **`mse`**: Computes the Mean Squared Error.
- **`mse_derivative`**: Computes the gradient of the MSE.

### `layer.py`

- Base class for defining layers, with placeholder methods for forward and backward passes.

### `activation.py`

- General framework for activation layers, enabling custom activation functions.

### `dense.py`

- Implements a fully connected layer with weight and bias initialization, forward propagation, and gradient updates.

## Visualization

The trained network's predictions for the XOR problem are visualized in 3D:

- X-axis and Y-axis: Inputs to the network.
- Z-axis: Predicted output.
- Color map: Intensity of the prediction.

## Contributing

Feel free to fork the repository and submit pull requests to enhance the project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

