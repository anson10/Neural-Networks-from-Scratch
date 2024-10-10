# Deep Learning Concepts

## Partial Derivatives
Partial derivatives are used in functions with multiple variables to understand how each variable influences the function. In deep learning, they are critical for optimizing the cost function.

- **Definition**: The partial derivative measures how a function changes as only one of its variables is varied, holding the others constant.
- **Notation**: For a function \( f(x, y) \), the partial derivative with respect to \( x \) is written as \( \frac{\partial f}{\partial x} \).
- **Example**: 
    Given \( f(x, y) = x^2 + 3xy + y^2 \), the partial derivatives would be:
    - \( \frac{\partial f}{\partial x} = 2x + 3y \)
    - \( \frac{\partial f}{\partial y} = 3x + 2y \)

- **Use in Deep Learning**: During training, partial derivatives are used to compute how the cost function changes with respect to weights and biases of the neural network.

## Gradient
The gradient generalizes partial derivatives to functions of multiple variables. It is a vector that contains all the partial derivatives of a function and points in the direction of the steepest increase of that function.

- **Definition**: The gradient of a function \( f(x_1, x_2, \ldots, x_n) \) is a vector of partial derivatives, i.e.,
  \[
  \nabla f = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right]
  \]
  
- **Role in Optimization**: In deep learning, the gradient is used to adjust weights in the opposite direction of the gradient to minimize the loss function. This is achieved through gradient descent.

- **Gradient Descent**: The weight update rule for gradient descent is given by:
  \[
  w = w - \eta \nabla J(w) 
  \]
  where:
    - \( w \) represents the weights,
    - \( \eta \) is the learning rate, and
    - \( \nabla J(w) \) is the gradient of the cost function \( J \) with respect to the weights.

- **Example**:
  Suppose we have a function \( f(x, y) = x^2 + y^2 \). The gradient is:
  \[
  \nabla f = [2x, 2y]
  \]

## Backpropagation
Backpropagation is the process of propagating the error (or loss) from the output layer back through the network to update the weights using the chain rule of calculus.

- **Forward Pass**: Inputs pass through the network, and outputs are calculated.
- **Loss Function**: The loss \( L \) is computed by comparing the network’s prediction with the actual target, e.g., using mean squared error (MSE):
  \[
  L = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  \]

- **Backward Pass**:
    - **Error Calculation**: Calculate the derivative of the loss function with respect to the output. 
    - **Chain Rule**: The chain rule is applied to compute the gradients of the loss function with respect to weights at each layer:
      \[
      \frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
      \]

    where:
      - \( \hat{y} \) is the predicted output,
      - \( z \) is the weighted sum of inputs at a neuron,
      - \( w \) is the weight.

- **Weight Update**:
  Once the gradients are calculated, the weights are updated as follows:
  \[
  w = w - \eta \frac{\partial L}{\partial w}
  \]

- **Chain Rule Example**: 
    Consider a two-layer network. For each weight \( w^{(1)} \) in the first layer, the gradient of the loss function with respect to this weight is:
    \[
    \frac{\partial L}{\partial w^{(1)}} = \frac{\partial L}{\partial a^{(2)}} \cdot \frac{\partial a^{(2)}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial w^{(1)}}
    \]

    This shows how the error is propagated back to earlier layers.

## Summary
Partial derivatives and gradients are used to calculate how the parameters (weights and biases) of a neural network should be adjusted to minimize the loss function. Backpropagation, combined with gradient descent, ensures that the weights are updated efficiently throughout the network by applying the chain rule.
