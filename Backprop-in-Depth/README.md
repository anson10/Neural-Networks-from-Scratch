## **Forward Pass through a Single Neuron with ReLU Activation**

In a neural network, the forward pass through a single neuron involves multiplying inputs (x) by corresponding weights (w) and adding a bias term (b). The result is then passed through an activation function, such as ReLU (Rectified Linear Unit), which outputs the final value y.

Mathematically, we represent the forward pass as:

$$ y = \text{ReLU}\left(\sum_{i=0}^{n} (x_i \cdot w_i) + b\right) $$

Where:
- \(x_i\) are the input values,
- \(w_i\) are the weights,
- \(b\) is the bias term,
- ReLU is the activation function defined as:
  
$$ \text{ReLU}(z) = \max(0, z) $$

The output \(y\) is the result of applying ReLU on the weighted sum of inputs and bias.

---

## **Backpropagation and Chain Rule**

In backpropagation, we calculate the gradients (partial derivatives) to update the weights and biases in the network. Since our neuron uses multiple functions (multiplication, addition, and ReLU), we apply the chain rule of differentiation to compute the overall derivative.

The chain rule for a composition of two functions \(f(g(x))\) is:

$$ \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x) $$

In the context of neural networks, this means we compute the derivative of each function in the chain and multiply them together. For example, in the expression:

$$ y = \text{ReLU}(x_0 w_0 + x_1 w_1 + x_2 w_2 + b) $$

We first need to compute the derivative of ReLU with respect to the sum:

$$ \frac{d}{d \text{sum}} \text{ReLU}(\text{sum}) $$

Then, we compute the derivative of the sum operation with respect to the specific weight \(w_0\):

$$ \frac{\partial}{\partial w_0} \left( x_0 w_0 + x_1 w_1 + x_2 w_2 + b \right) = x_0 $$

Thus, the full derivative for \(w_0\) is:

$$ \frac{\partial}{\partial w_0} \left[ \text{ReLU}\left(\sum_{i=0}^{n} (x_i \cdot w_i) + b\right) \right] 
= \frac{d \text{ReLU}}{d \text{sum}} \cdot \frac{\partial \text{sum}}{\partial w_0} = \text{ReLU}'(\text{sum}) \cdot x_0 $$

---

## **ReLU Derivative**

The derivative of the ReLU function is straightforward:

$$ \text{ReLU}(z) = \max(0, z) $$

The derivative of ReLU with respect to its input \(z\) is:

- \(1\) if \(z > 0\),
- \(0\) if \(z \leq 0\).

In mathematical notation:

$$ \frac{d}{dz} \text{ReLU}(z) = 1(z > 0) $$

This can be written in Python as:

```python
relu_dz = (1. if z > 0 else 0.)
```

For example, if the input to ReLU is \(6\), then:

$$ \text{ReLU}'(6) = 1 $$

We multiply this result by the derivative from the next layer (in our case, it’s assumed to be 1 for simplicity):

$$ \delta = 1 \cdot 1 = 1 $$

---

## **Full Derivative Calculation**

Using the chain rule, the derivative with respect to \(x_0\) is:

$$ \frac{\partial}{\partial x_0} \left[ \text{ReLU} \left( \sum_{i=0}^{n} (x_i \cdot w_i) + b \right) \right] = \frac{d \text{ReLU}}{d \text{sum}} \cdot \frac{\partial \text{sum}}{\partial x_0} $$

This derivative informs us about the impact of \(x_0\) on the output. The same process can be repeated to calculate the impact of weights and biases.

---

## **Backward Pass: Gradient Calculation**

In the backward pass, gradients flow back through the network. The derivative of the loss function with respect to each weight and bias is calculated, and these gradients are used to update the parameters. The chain rule is applied at each layer, allowing us to propagate the error backward through the network.

- The gradient of the loss with respect to weights updates the weights,
- The gradient with respect to biases updates the biases,
- The gradient with respect to inputs is propagated to the previous layer.

---

This breakdown provides a structured approach to understanding backpropagation through a neuron using ReLU activation.

# Categorical Cross-Entropy Loss

**Categorical Cross-Entropy Loss** is used in classification problems to measure how well the predicted probabilities match the true one-hot encoded labels. It is commonly used in multi-class classification tasks.

## Formula

The formula for **Categorical Cross-Entropy Loss** is:

$$
L = - \sum_{i=1}^{N} y_i \cdot \log(\hat{y}_i)
$$

Where:
- $L$ is the total loss.
- $y_i$ is the true label for class $i$, which is usually one-hot encoded. This means that $y_i = 1$ for the correct class, and $y_i = 0$ for the other classes.
- $\hat{y}_i$ is the predicted probability for class $i$.
- $N$ is the number of classes.

## Explanation

The **Categorical Cross-Entropy Loss** works by comparing the predicted probabilities $\hat{y}_i$ to the true labels $y_i$. Here's how it works step by step:

1. For each class $i$, the true label $y_i$ is either 0 or 1 (because it's one-hot encoded).
2. The logarithm of the predicted probability $\log(\hat{y}_i)$ is taken for the class $i$.
3. This value is multiplied by the true label $y_i$ (which ensures only the correct class contributes to the loss).
4. The sum of all class contributions is then negated to get the final loss.

### Key Points:
- If the predicted probability for the true class is high (close to 1), the loss is small.
- If the predicted probability for the true class is low (close to 0), the loss is large.
- The objective of training is to minimize this loss, making the model’s predicted probabilities closer to the true one-hot labels.

## Why Use It?

- **Penalizes incorrect predictions**: By taking the logarithm, incorrect or low-confidence predictions are heavily penalized.
- **Smooth optimization**: The cross-entropy loss is differentiable, which makes it useful for gradient-based optimization techniques such as backpropagation in neural networks.
