# Binary Logistic Regression

Binary Logistic Regression is a type of regression analysis used when the dependent variable (also called the target or outcome) is **binary** (i.e., it can take two possible values). It is used to model the probability that a given input belongs to one of the two possible classes.

## Key Concepts

1. **Binary Outcome**: The dependent variable is binary, typically represented as:
   - 0 or 1
   - True or False
   - Yes or No

2. **Logistic Function**: Binary logistic regression uses the **logistic function** (also called the **sigmoid function**) to map the input values to probabilities between 0 and 1.

   The logistic function is defined as:

   $$
   \sigma(z) = \frac{1}{1 + e^{-z}}
   $$

   Where:
   - $ \sigma(z) $ is the predicted probability.
   - $ z $ is the linear combination of the input features and their weights.

3. **Model Equation**: The logistic regression model estimates the probability that the dependent variable equals 1 (positive class). The model can be written as:

   $$
   P(y = 1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}
   $$

   Where:
   - $ P(y = 1 | X) $ is the probability that the output is 1 given the input $ X $.
   - $ \beta_0 $ is the intercept (bias term).
   - $ \beta_1, \dots, \beta_n $ are the coefficients (weights) for the input features $ x_1, \dots, x_n $.

4. **Interpretation of Coefficients**:
   - The coefficients $ \beta_1, \beta_2, \dots $ represent the effect of each input feature on the log-odds of the outcome.
   - The log-odds (or the logit) is the logarithm of the ratio of the probability of the positive class to the probability of the negative class.

   $$
   \text{log-odds} = \ln\left(\frac{P(y=1)}{P(y=0)}\right)
   $$

5. **Decision Boundary**:
   - A probability threshold is set to classify the predicted output as either 0 or 1.
   - The most common threshold is 0.5:
     - If $ P(y = 1 | X) \geq 0.5 $, predict class 1.
     - If $ P(y = 1 | X) < 0.5 $, predict class 0.

## Training Process

1. **Maximum Likelihood Estimation (MLE)**: Logistic regression uses MLE to estimate the best-fitting coefficients. It maximizes the likelihood of the observed data given the model.

2. **Cost Function (Log Loss)**: The loss function used in logistic regression is the **logarithmic loss** (also called binary cross-entropy), defined as:

   $$
   \text{Cost} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
   $$

   Where:
   - $ m $ is the number of training examples.
   - $ y^{(i)} $ is the actual label for the $ i $-th example.
   - $ \hat{y}^{(i)} $ is the predicted probability for the $ i $-th example.

# Sigmoid Activation Function and its Derivative

## Sigmoid Function

The **sigmoid function** is commonly used in binary classification tasks, as it maps any real-valued number to a value between 0 and 1, which can be interpreted as a probability.

The sigmoid function is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where:
- $z$ is the input.
- $e$ is Euler's number (approximately 2.718).

## Properties of the Sigmoid Function

- Output range: $(0, 1)$
- Smooth and differentiable.
- Symmetrical around $z = 0$.
- As $z \to \infty$, $\sigma(z) \to 1$.
- As $z \to -\infty$, $\sigma(z) \to 0$.

## Derivative of Sigmoid Function

The derivative of the sigmoid function is important for backpropagation in neural networks. Let's derive it step by step:

### Step-by-Step Derivation

Starting with the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

We need to find the derivative of $\sigma(z)$ with respect to $z$.

1. **Rewrite the sigmoid function**:
   We first express the function in a more convenient form for differentiation:

   $$
   \sigma(z) = (1 + e^{-z})^{-1}
   $$

2. **Differentiate using the chain rule**:
   Apply the chain rule to find the derivative of $\sigma(z)$:

   $$
   \frac{d}{dz} \sigma(z) = \frac{d}{dz} \left( (1 + e^{-z})^{-1} \right)
   $$

   Using the chain rule:

   $$
   \frac{d}{dz} \left( (1 + e^{-z})^{-1} \right) = -(1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})
   $$

3. **Differentiate the inner expression**:
   Now, differentiate $1 + e^{-z}$:

   $$
   \frac{d}{dz}(1 + e^{-z}) = -e^{-z}
   $$

4. **Substitute back into the chain rule**:
   Substitute the result into the chain rule expression:

   $$
   \frac{d}{dz} \sigma(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z})
   $$

5. **Simplify**:
   The negatives cancel out, leaving us with:

   $$
   \frac{d}{dz} \sigma(z) = \frac{e^{-z}}{(1 + e^{-z})^2}
   $$

6. **Relate to the sigmoid function**:
   Now, express this result in terms of $\sigma(z)$. Recall that:

   $$
   \sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{and} \quad 1 - \sigma(z) = \frac{e^{-z}}{1 + e^{-z}}
   $$

   Therefore, the derivative can be written as:

   $$
   \frac{d}{dz} \sigma(z) = \sigma(z) \cdot (1 - \sigma(z))
   $$

### Final Result

The derivative of the sigmoid function is:

$$
\frac{d}{dz} \sigma(z) = \sigma(z) \cdot (1 - \sigma(z))
$$

This result is commonly used in backpropagation algorithms, making it efficient to calculate the gradient during training of neural networks.

# Binary Cross-Entropy Loss and Its Derivative

## What is Binary Cross-Entropy Loss?

Binary cross-entropy (also known as log loss) is used for binary classification tasks. It measures the difference between two probability distributions — the predicted probability and the actual label (0 or 1). 

The formula for **binary cross-entropy loss** is:

$$
L = - \frac{1}{m} \sum_{i=1}^{m} \left( y_i \cdot \log(\hat{y}_i) + (1 - y_i) \cdot \log(1 - \hat{y}_i) \right)
$$

Where:
- $m$ is the number of training examples.
- $y_i$ is the true label (either 0 or 1).
- $\hat{y}_i$ is the predicted probability that the output is 1, $\hat{y}_i = \sigma(z_i)$.
- $\log$ is the natural logarithm.

This loss penalizes incorrect predictions. If the predicted probability ($\hat{y}_i$) is far from the actual label ($y_i$), the loss will be higher.

### Intuition

- If $y_i = 1$: The loss focuses on $\log(\hat{y}_i)$, so the predicted probability $\hat{y}_i$ should be close to 1 to minimize the loss.
- If $y_i = 0$: The loss focuses on $\log(1 - \hat{y}_i)$, so the predicted probability $\hat{y}_i$ should be close to 0 to minimize the loss.

## Derivative of Binary Cross-Entropy Loss

The **derivative of binary cross-entropy loss** with respect to the predicted probability $\hat{y}_i$ is needed for backpropagation in neural networks. Let’s derive it step-by-step.

### Loss Function for a Single Example

The binary cross-entropy loss for a single example can be written as:

$$
L(y, \hat{y}) = - \left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)
$$

### Taking the Derivative

To find the derivative of the loss with respect to the predicted probability $\hat{y}$:

1. **Use the chain rule**:
   We need to differentiate the loss function:

   $$
   \frac{dL}{d\hat{y}} = -\left( \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right)
   $$

2. **Simplify**:
   This can be simplified to:

   $$
   \frac{dL}{d\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}
   $$

### Final Result

The derivative of the binary cross-entropy loss is:

$$
\frac{dL}{d\hat{y}} = \frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}
$$

This derivative is useful for updating the weights during the training process of a neural network using gradient descent.

