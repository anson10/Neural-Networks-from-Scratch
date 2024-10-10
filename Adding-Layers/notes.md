## Hidden Layers in Neural Networks

In a neural network, hidden layers are the layers between the input and output layers. They process input data through learned transformations.

### Structure

1. **Input Layer**: Receives the raw data.
2. **Hidden Layers**: Apply transformations to the data.
3. **Output Layer**: Provides the final prediction.

### Components

- **Neurons**: Perform computations.
- **Weights and Biases**: Adjust the inputs.
- **Activation Function**: Adds non-linearity.

#### Mathematical Representation
-------------------------------

For a single neuron in a hidden layer, the output is given by:

$$a_j = \sigma\left(\sum_{i} w_{ij} x_i + b_j\right)$$

where:

- $a_j$ is the output,
- $\sigma$ is the activation function,
- $w_{ij}$ is the weight,
- $x_i$ is the input,
- $b_j$ is the bias.


In a multi-layer network, the output of a neuron in layer $l$ is:

$$a_j^{(l)} = \sigma\left(\sum_{i} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}\right)$$

where:

- $a_j^{(l)}$ is the output in layer $l$,
- $w_{ij}^{(l)}$ is the weight connecting neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$,
- $b_j^{(l)}$ is the bias for neuron $j$ in layer $l$.

# NNFS: Neural Network from Scratch Package

## Overview

NNFS (Neural Network from Scratch) is a Python package designed to facilitate the creation and implementation of neural networks from scratch.


## Installation

```bash
pip install nnfs
```

## Key Features

### 1. Dataset Generation
-------------------------

NNFS includes functions to generate synthetic datasets for testing and training neural networks. One such function is spiral_data, which creates a non-linear dataset with customizable number of classes and points per class.

### 2. Initialization
The nnfs.init() function ensures repeatability of results by:
Setting the random seed to 0 (default)
Setting the default data type to float32
Overriding the original NumPy dot product

# Layer_Dense Class Overview

## Introduction

The `Layer_Dense` class represents a dense (fully-connected) layer in a neural network. Each neuron in this layer is connected to every neuron in the previous layer. This class includes methods for initializing the layer and performing a forward pass.

## Class Definition

### `__init__` Method

The `__init__` method initializes the weights and biases for the layer.

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
```
## Weights Initialization:

- `np.random.randn(n_inputs, n_neurons)` generates random weights from a Gaussian distribution.
- Multiplying by 0.01 scales the weights to smaller values, preventing issues during training.

## Biases Initialization:

- `np.zeros((1, n_neurons))` initializes biases to zero. Zero is a common choice, but in some cases, biases may be initialized to non-zero values.

## `forward` Method:
- The forward method performs the forward pass through the layer, calculating the output values.

```python
def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
```

### Dot Product

- `np.dot(inputs, self.weights)` computes the dot product between the input data and the weights.

### Adding Biases:

- `+ self.biases` adds the biases to the dot product result.

```python
import numpy as np
import nnfs
nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Print output of the first few samples
print(dense1.output[:5])
```

## Summary

- **Initialization:** Weights are initialized with small random values and biases are initialized to zero.
- **Forward Pass:** Computes the output by taking the dot product of inputs and weights and adding biases.
- **Example:** Demonstrates how to create a dataset, initialize a dense layer, and perform a forward pass.