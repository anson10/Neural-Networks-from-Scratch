# Deep Learning Notes: Derivatives

## 1. Slopes and Derivatives

### Concept of a Slope:
- The **slope** of a function at a given point describes how the function's output changes with respect to a change in its input.
- In mathematical terms, the slope is defined as the rate of change of a function `f(x)` with respect to the variable `x`, and it is represented as the derivative `df/dx`.

### Derivative Definition:
- The derivative of a function `f(x)` at a point `x = a` is given by:

  `f'(a) = lim_{Δx → 0} (f(a + Δx) - f(a)) / Δx`

This formula provides the **instantaneous rate of change** at point `a`.

---

## 2. Numerical Derivation

### Numerical Approximation of Derivatives:
- In practical situations (like deep learning), exact analytical derivation may be complex or impossible. In such cases, numerical methods are used to approximate derivatives.

### Forward Difference Formula:
- A common way to numerically approximate the derivative is by using the **forward difference method**:

  `f'(x) ≈ (f(x + h) - f(x)) / h`

where `h` is a small value (e.g., `h = 10^-5`).

### Central Difference Formula:
- A more accurate method is the **central difference** formula, which is calculated using points on both sides of `x`:

  `f'(x) ≈ (f(x + h) - f(x - h)) / (2h)`

This method reduces numerical error compared to the forward difference method.

### Higher-order Derivatives:
- Second derivative approximation:

  `f''(x) ≈ (f(x + h) - 2f(x) + f(x - h)) / h^2`

Higher-order derivatives can be similarly approximated using finite difference methods.

---

## 3. Analytical Derivation

### Basic Rules for Derivatives:
1. **Power Rule**: 
   
   `d/dx [x^n] = nx^(n-1)`
   
   Example: For `f(x) = x^3`, the derivative is `f'(x) = 3x^2`.

2. **Sum Rule**: 
   
   `d/dx [f(x) + g(x)] = f'(x) + g'(x)`

3. **Product Rule**:
   
   `d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)`

4. **Quotient Rule**:
   
   `d/dx [f(x)/g(x)] = (f'(x)g(x) - f(x)g'(x)) / [g(x)]^2`

5. **Chain Rule**:
   - Used when dealing with **composite functions**:
   
   `d/dx f(g(x)) = f'(g(x)) · g'(x)`
   
   Example: For `f(x) = sin(x^2)`, `f'(x) = cos(x^2) · 2x`.

### Common Derivatives:
- Derivative of `e^x`: 
  
  `d/dx [e^x] = e^x`
  
- Derivative of `sin(x)`: 
  
  `d/dx [sin(x)] = cos(x)`

- Derivative of `ln(x)`: 
  
  `d/dx [ln(x)] = 1/x`

### Gradient in Multi-Dimensional Functions:
- In deep learning, we often deal with multi-dimensional functions. The **gradient** is the vector of partial derivatives with respect to all input variables:

  `∇f(x_1, x_2, ..., x_n) = [∂f/∂x_1, ∂f/∂x_2, ..., ∂f/∂x_n]`

The gradient points in the direction of the steepest ascent.

---

## 4. Application of Derivatives in Deep Learning

### Backpropagation:
- Derivatives are crucial in **backpropagation**, the method used to compute gradients in neural networks.
- The error function is minimized by adjusting weights using the gradient of the error with respect to the weights.

### Gradient Descent:
- **Gradient Descent** is an optimization algorithm used to minimize the loss function in neural networks. The update rule for weights is:

  `w = w - η ∇L(w)`

where `η` is the learning rate and `∇L(w)` is the gradient of the loss function with respect to the weights.

---

### Summary:
- **Slopes and derivatives** provide a mathematical framework for measuring changes in functions.
- **Numerical derivatives** offer an approximate solution when analytical derivation is not feasible.
- **Analytical derivatives** use rules like the power rule, product rule, chain rule, etc., to compute exact changes.
- In deep learning, derivatives are used in **gradient descent** and **backpropagation** for training models.

---