# Regularization

Regularization is a technique used to prevent overfitting in machine learning models by adding a **penalty** to the loss function. Overfitting occurs when a model performs well on training data but poorly on unseen data.

### Types of Regularization:

1. **L1 Regularization (Lasso)**:
   - Adds the **absolute value** of the weights to the loss function.
   - Encourages sparsity in the model, meaning some weights are reduced to zero, effectively performing feature selection.

   **Penalty Term:**
    - $\lambda \sum |w|$


2. **L2 Regularization (Ridge)**:
- Adds the **squared value** of the weights to the loss function.
- Helps to distribute the error across the weights, keeping all features but reducing their impact.

**Penalty Term:**
- $\lambda \sum w^2$

### Effect of Regularization:
- **Reduces variance** in the model by penalizing large coefficients, which leads to better generalization on unseen data.
- The **regularization strength** is controlled by the hyperparameter **λ** (lambda). A larger λ increases the penalty and reduces the complexity of the model.

### Trade-off:
- Regularization introduces a **bias-variance trade-off**: increasing λ reduces variance (overfitting) but increases bias, potentially underfitting the model.

# Forward Pass with Regularization

In the forward pass of a neural network, data is passed through the layers, where each layer applies its learned weights and biases to generate an output. Regularization is applied to prevent the model from overfitting by adding a penalty to the loss function.

### L1 Regularization:

L1 regularization adds a penalty proportional to the **sum of the absolute values** of the weights and biases. The penalty term for L1 regularization can be expressed as:

$$
L1_{\text{penalty}} = \lambda \sum_{i} |w_i| + \lambda \sum_{i} |b_i|
$$

Where:
- $w_i$ are the weights
- $b_i$ are the biases
- $\lambda$ is the regularization parameter that controls the strength of the penalty.

Since this is a linear penalty, L1 regularization has the effect of driving many weights towards zero, encouraging sparsity in the model. However, it may also affect small weights more aggressively, making the model less sensitive to small input features.

### L2 Regularization:

L2 regularization, on the other hand, adds a penalty based on the **sum of the squared values** of the weights and biases. This penalty term is given by:

$$
L2_{\text{penalty}} = \lambda \sum_{i} w_i^2 + \lambda \sum_{i} b_i^2
$$

In this case, larger weights are penalized more because the square of a large value increases significantly, while small weights are only slightly penalized. This helps prevent the weights from becoming too large without strongly affecting small parameter values.

### Regularized Loss Function:

The total loss function for the network with regularization includes both the data loss and the regularization penalty. For a loss function $L$ (e.g., mean squared error, cross-entropy), the regularized loss is:

$$
\text{Total Loss} = L(\text{output}, \text{target}) + \text{Regularization Penalty}
$$

For L1 regularization, this becomes:

$$
\text{Total Loss} = L(\text{output}, \text{target}) + \lambda \sum_{i} |w_i| + \lambda \sum_{i} |b_i|
$$

For L2 regularization, the loss function is:

$$
\text{Total Loss} = L(\text{output}, \text{target}) + \lambda \sum_{i} w_i^2 + \lambda \sum_{i} b_i^2
$$

### Combining L1 and L2 (Elastic Net):

In some cases, both L1 and L2 penalties are applied simultaneously. This combination is called **Elastic Net Regularization**, and the penalty term is:

$$
\text{Elastic Net Penalty} = \alpha \left( \lambda_1 \sum_{i} |w_i| + \lambda_2 \sum_{i} w_i^2 \right)
$$

Where $\alpha$ controls the balance between L1 and L2 regularization.

### Impact of Regularization:

Regularization drives the weights and biases towards smaller values (ideally close to 0). This helps with model stability, especially in cases of **exploding gradients** where weights can become excessively large.

The strength of the regularization penalty is controlled by the hyperparameter **$\lambda$**, where:
- **Higher $\lambda$** results in stronger regularization, which can lead to underfitting.
- **Lower $\lambda$** applies a weaker penalty, potentially allowing overfitting if the model is too complex.

---

# Backward Pass

The backward pass in neural networks is essential for updating the weights during training. It involves calculating the gradients of the loss function with respect to the weights. Regularization techniques, such as L1 and L2 regularization, influence these gradients to help prevent overfitting.

## L2 Regularization

The derivative of L2 regularization is straightforward:

$$
L_{2w} = \lambda \sum_{m} w_{m}^2 \implies \frac{\partial L_{2w}}{\partial w_{m}} = \lambda \frac{\partial}{\partial w_{m}} \left[ \sum_{m} w_{m}^2 \right] = \lambda \cdot 2w_{m} = 2\lambda w_{m}
$$

- Here, $\lambda$ is a constant (the regularization strength).
- The derivative calculation shows that L2 regularization increases the penalty for larger weights, effectively encouraging smaller weights through the gradient update.

This calculation simplifies to multiplying the weights by $2\lambda$, which can be efficiently implemented using NumPy.

## L1 Regularization

The derivative for L1 regularization is more complex due to the nature of the absolute value function:

$$
f(x) =
\begin{cases}
x & \text{if } x > 0 \\
-x & \text{if } x < 0
\end{cases} 
\implies f'(x) =
\begin{cases}
1 & \text{if } x > 0 \\
-1 & \text{if } x < 0
\end{cases}
$$

For L1 regularization, the complete partial derivative with respect to a given weight is:

$$
L_{1w} = \lambda \sum_{m} |w_{m}| \implies L'_{1w} = \frac{\partial L_{1w}}{\partial w_{m}} = \lambda \frac{\partial}{\partial w_{m}} |w_{m}| = 
\begin{cases}
\lambda & \text{if } w_{m} > 0 \\
-\lambda & \text{if } w_{m} < 0
\end{cases}
$$

- In this case, the derivative can equal $1$ or $-1$ depending on the sign of the weight $w_{m}$. 

### Summary

- **L2 Regularization**: Penalizes larger weights by scaling their gradient, leading to smaller weights over time.
- **L1 Regularization**: Provides a piecewise derivative that promotes sparsity in the weights, encouraging many weights to be zero.

Understanding these derivatives is crucial for effectively implementing gradient descent and ensuring that the model does not overfit the training data.
