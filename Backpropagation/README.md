# Deep Learning Concepts

## Partial Derivatives
Partial derivatives are used in functions with multiple variables to understand how each variable influences the function. In deep learning, they are critical for optimizing the cost function.

- **Definition**: The partial derivative measures how a function changes as only one of its variables is varied, holding the others constant.
- **Notation**: For a function f(x, y), the partial derivative with respect to x is written as ∂f/∂x.
- **Example**: 
    Given f(x, y) = x^2 + 3xy + y^2, the partial derivatives would be:
    - ∂f/∂x = 2x + 3y
    - ∂f/∂y = 3x + 2y

- **Use in Deep Learning**: During training, partial derivatives are used to compute how the cost function changes with respect to weights and biases of the neural network.

## Gradient
The gradient generalizes partial derivatives to functions of multiple variables. It is a vector that contains all the partial derivatives of a function and points in the direction of the steepest increase of that function.

- **Definition**: The gradient of a function f(x1, x2, ..., xn) is a vector of partial derivatives:
  ∇f = [ ∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn ]
  
- **Role in Optimization**: In deep learning, the gradient is used to adjust weights in the opposite direction of the gradient to minimize the loss function. This is achieved through gradient descent.

- **Gradient Descent**: The weight update rule for gradient descent is given by:
  w = w - η ∇J(w) 
  where:
    - w represents the weights,
    - η is the learning rate, and
    - ∇J(w) is the gradient of the cost function J with respect to the weights.

- **Example**:
  Suppose we have a function f(x, y) = x^2 + y^2. The gradient is:
  ∇f = [2x, 2y]

## Backpropagation
Backpropagation is the process of propagating the error (or loss) from the output layer back through the network to update the weights using the chain rule of calculus.

- **Forward Pass**: Inputs pass through the network, and outputs are calculated.
- **Loss Function**: The loss L is computed by comparing the network’s prediction with the actual target, e.g., using mean squared error (MSE):
  L = (1/n) * Σ(y_i - ŷ_i)^2

- **Backward Pass**:
    - **Error Calculation**: Calculate the derivative of the loss function with respect to the output. 
    - **Chain Rule**: The chain rule is applied to compute the gradients of the loss function with respect to weights at each layer:
      ∂L/∂w = (∂L/∂ŷ) * (∂ŷ/∂z) * (∂z/∂w)

    where:
      - ŷ is the predicted output,
      - z is the weighted sum of inputs at a neuron,
      - w is the weight.

- **Weight Update**:
  Once the gradients are calculated, the weights are updated as follows:
  w = w - η (∂L/∂w)

- **Chain Rule Example**: 
    Consider a two-layer network. For each weight w^(1) in the first layer, the gradient of the loss function with respect to this weight is:
    
    ∂L/∂w^(1) = (∂L/∂a^(2)) * (∂a^(2)/∂z^(2)) * (∂z^(2)/∂a^(1)) * (∂a^(1)/∂z^(1)) * (∂z^(1)/∂w^(1))

    This shows how the error is propagated back to earlier layers.

## Summary
Partial derivatives and gradients are used to calculate how the parameters (weights and biases) of a neural network should be adjusted to minimize the loss function. Backpropagation, combined with gradient descent, ensures that the weights are updated efficiently throughout the network by applying the chain rule.
