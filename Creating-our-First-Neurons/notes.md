# Chapter 1: Coding Our First Neurons

## Inputs, Weights, and Biases

In a neural network, a neuron takes **inputs**, processes them by applying **weights** and **biases**, and then produces an output. The relationship can be described as:

### Inputs
Inputs are the data that a neuron receives. In the simplest case, inputs can be numerical values representing features of a dataset. For example, in a neural network designed to classify images, inputs might be pixel values.

### Weights
Weights determine the importance of each input to the neuron's output. Each input has an associated weight, and the weighted sum of all inputs is calculated. The larger the weight, the more influence that input has on the final output.

\[
\text{Weighted Sum} = (input_1 \times weight_1) + (input_2 \times weight_2) + ... + (input_n \times weight_n)
\]

### Biases
Bias is an additional value added to the weighted sum. It helps the neuron make adjustments beyond the combination of inputs and weights, allowing the model to fit the data better.

\[
\text{Output} = \text{ActivationFunction}(\text{Weighted Sum} + \text{Bias})
\]

### Example:
```python
inputs = [1.2, 3.5, 5.1]
weights = [0.4, 0.8, 0.6]
bias = 2

# Weighted sum of inputs + bias
output = (inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + bias
print(output)

# Layer of Neurons

In a neural network, neurons are typically organized in **layers**. A **layer** is a group of neurons that work together to process the same input, each with its own set of weights and biases. The outputs from one layer become the inputs for the next layer, allowing the network to build increasingly complex representations of the data.

## Fully Connected (Dense) Layer

In a **fully connected layer** (also called a dense layer), every neuron is connected to all the neurons from the previous layer. This means each neuron receives the same set of inputs but applies its own weights and bias to compute the output.

### Key Components:

- **Inputs:** Data that the neurons receive, typically the outputs from the previous layer.
- **Weights:** Each neuron has its own set of weights that scale the importance of each input.
- **Biases:** Bias allows the neuron to adjust the output independently of the inputs, giving it additional flexibility.
  
### Why Multiple Neurons?

A layer can have multiple neurons to allow the network to capture a range of different patterns from the same input. Each neuron focuses on different aspects of the input data, and together, they provide a more comprehensive analysis. 

For example, in image recognition, some neurons might learn to detect edges, while others might detect more complex shapes.

## Example: Layer with 3 Neurons

Let’s say we have a layer with 3 neurons, each with its own weights and biases. Here’s how we can compute the outputs for this layer:

```python
import numpy as np

# Inputs from the previous layer
inputs = [1.2, 3.5, 5.1]

# Weights for 3 neurons, each with 3 inputs
weights = [
    [0.2, 0.8, -0.5],  # Weights for Neuron 1
    [0.5, -0.91, 0.26], # Weights for Neuron 2
    [-0.26, -0.27, 0.17] # Weights for Neuron 3
]

# Biases for each neuron
biases = [2, 3, 0.5]

# Output of the layer (dot product of inputs and weights + bias)
layer_output = np.dot(weights, inputs) + biases

print(layer_output)
```

# Layer of Neurons and Batch of Data with NumPy

In this section, we’ll explore how to process a **layer of neurons** using a **batch of data** with the help of NumPy. This concept is crucial for efficiently handling multiple samples of data at once, a common requirement in neural networks.

## Layer of Neurons

A **layer** consists of multiple neurons, each of which processes the same inputs but applies different weights and biases. In a neural network, each neuron in the layer receives the same input and produces an output that is passed to the next layer.

### Example: A Layer with 3 Neurons

```python
import numpy as np

# Inputs from the previous layer (for a single data point)
inputs = [1.2, 3.5, 5.1]

# Weights for 3 neurons, each neuron has 3 inputs
weights = [
    [0.2, 0.8, -0.5],  # Weights for Neuron 1
    [0.5, -0.91, 0.26], # Weights for Neuron 2
    [-0.26, -0.27, 0.17] # Weights for Neuron 3
]

# Biases for each neuron
biases = [2, 3, 0.5]

# Output of the layer (dot product of inputs and weights + bias)
layer_output = np.dot(weights, inputs) + biases

print(layer_output)
```