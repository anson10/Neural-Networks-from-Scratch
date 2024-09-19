Activation function

Step Activation Function
Linear Activation Function
Sigmoid Activation Function
RELU
Softmax Activation Function


Real Use for Activation Function


# Activation Functions Overview

## What are Activation Functions?

Activation functions introduce non-linearity into neural networks, enabling them to learn complex relationships.


## Common Activation Functions

| Function        | Description                  | Formula                         |
| -------------- | ---------------------------- | ------------------------------- |
| **Sigmoid**     | Maps inputs to (0, 1)        | σ(x) = 1 / (1 + e^(-x))        |
| **ReLU**        | Maps negative inputs to 0    | f(x) = max(0, x)               |
| **Step**        | Binary activation (0 or 1)   | f(x) = 1 if x ≥ 0, 0 otherwise |
| **Linear**      | Identity function            | f(x) = x                       |
| **Softmax**     | Normalizes inputs for probability distribution | σ(x) = e^x / Σ(e^x) |


## Example Use Case

### Code
```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(0, x)

def step(x):
  return np.where(x >= 0, 1, 0)

def linear(x):
  return x

# Example input
x = np.array([1, 2, 3])

# Apply activation functions
sigmoid_output = sigmoid(x)
relu_output = relu(x)
step_output = step(x)
linear_output = linear(x)

print("Sigmoid Output:", sigmoid_output)
print("ReLU Output:", relu_output)
print("Step Output:", step_output)
print("Linear Output:", linear_output)
```
# Step Activation Function

The Step Activation Function, also known as the Heaviside Step Function or Unit Step Function, is a simple activation function used in neural networks. It is a binary activation function that outputs either 0 or 1, depending on the input.


## Mathematical Representation

### Formula
## 1. Step Activation Function

The **Step Activation Function** is a binary activation function where the output is either `0` or `1`. It is primarily used for binary classification. When the input value is below a certain threshold (usually 0), the output is 0, and when the input value exceeds the threshold, the output is 1.

### Formula:
$$
f(x) =
\begin{cases}
0 & \text{if } x < \theta \\
1 & \text{if } x \geq \theta
\end{cases}
$$

## 2. Linear Activation Function

The **Linear Activation Function** is the simplest type of activation function, where the output is directly proportional to the input. It is primarily used in simpler machine learning models like linear regression, but it is rarely used in deep neural networks due to its limitations.

## Formula

The formula for the linear activation function is:

f(x) = x

This means that the output is exactly the same as the input — there is no transformation or non-linearity applied.

# ReLU Activation Function

## Introduction

The **ReLU (Rectified Linear Unit)** activation function is one of the most popular and widely used activation functions in deep learning. It introduces non-linearity into the model while being computationally efficient, making it a common choice for the hidden layers of neural networks.

## Formula

The formula for the ReLU activation function is:

$
f(x) = \max(0, x)
$

This means that the output is equal to the input if the input is positive, and 0 otherwise.

### Mathematical Representation:
- If $x > 0$, then $f(x) = x$
- If $x \leq 0$, then $f(x) = 0$

# Softmax Activation Function

## Introduction

The **Softmax Activation Function** is primarily used in the output layer of classification problems, especially in multi-class classification. It converts raw model outputs (logits) into probabilities, ensuring the output values sum up to 1. This allows the model to interpret the output as a probability distribution over multiple classes.

## Formula

For a vector of inputs $\mathbf{z} = [z_1, z_2, \dots, z_n]$, the Softmax function is calculated as:

$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$


Where:
- $z_i$ is the $i$-th element of the input vector.
- $e^{z_i}$ is the exponential of the $i$-th element.
- The denominator sums the exponentials of all elements in the input vector.

The result is a probability distribution where each value is between 0 and 1, and the sum of all the probabilities equals 1.

## Example

Given an input vector $\mathbf{z} = [z_1, z_2, z_3] = [2, 1, 0]$:

1. Calculate the exponentials:
   $e^{z_1} = e^2$, $e^{z_2} = e^1$, $e^{z_3} = e^0 = 1$

2. Sum of exponentials:
   $e^2 + e^1 + e^0 = e^2 + e + 1$

3. Apply the Softmax function to each element:\
   $\sigma(\mathbf{z})_1 = \frac{e^2}{e^2 + e + 1}$, \
   $\sigma(\mathbf{z})_2 = \frac{e^1}{e^2 + e + 1}$, \
   $\sigma(\mathbf{z})_3 = \frac{1}{e^2 + e + 1}$


The output will be a vector of probabilities corresponding to each class.

## Characteristics

1. **Probabilities as Output**: The Softmax function converts raw scores (logits) into probabilities that can be interpreted as the likelihood of each class. This makes it particularly useful for classification problems.

2. **Multi-class Classification**: It is widely used in problems where we need to assign probabilities across multiple classes (e.g., digit recognition where the output can be one of 10 digits).

3. **Normalization**: The Softmax function ensures that the sum of all the outputs is 1, making the outputs suitable as a probability distribution.

## Use Cases

Softmax is primarily used in:
- **Multi-class classification** tasks (e.g., image classification).
- The **output layer of neural networks** where the task is to assign one input to one of several categories.

## Limitations

1. **Exponential Sensitivity**: Since the Softmax function uses exponentials, large input values can lead to numerical instability (overflow issues). However, this can often be mitigated by subtracting the maximum value of the input vector from all elements before applying Softmax.

2. **Overconfidence in Predictions**: Softmax can sometimes output very high probabilities for certain classes, leading to overconfident predictions, even if the actual confidence is lower.

## Conclusion

The Softmax activation function is essential for transforming logits into probabilities, especially in multi-class classification problems. It ensures that the output of a neural network can be interpreted as a probability distribution, which is crucial for many classification tasks.




