# Categorical Cross-Entropy Loss

## Introduction

Categorical cross-entropy loss is a widely used loss function in multi-class classification problems. It quantifies the difference between the predicted probabilities and the true distribution of classes. This loss function is particularly useful when dealing with one-hot encoded labels.

## Mathematical Formulation

### Definition

Given a model's predicted probabilities $\hat{y}$ for $K$ classes and the true distribution $y$ (which is often represented as a one-hot encoded vector), the categorical cross-entropy loss can be defined mathematically as follows:

$$
L(y, \hat{y}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)
$$

Where:
- $L(y, \hat{y})$ is the loss.
- $y_i$ is the true probability of class $i$ (1 for the true class, 0 for all others in the case of one-hot encoding).
- $\hat{y}_i$ is the predicted probability of class $i$.

### Explanation of the Formula

1. **Logarithm**: The logarithm function measures how "surprised" we are by the predicted probability of the true class. If $\hat{y}_i$ is high (i.e., the model is confident), the log value will be close to 0, resulting in a smaller loss. If $\hat{y}_i$ is low (i.e., the model is uncertain), the log value will be more negative, leading to a larger loss.

2. **One-Hot Encoding**: In one-hot encoding, only one element of $y$ is 1 (indicating the true class), and all other elements are 0. This simplifies our loss calculation to:
   $$
   L(y, \hat{y}) = -\log(\hat{y}_j)
   $$
   where $j$ is the index of the true class.

3. **Summation Over Classes**: The loss is computed over all classes, but due to the one-hot encoding, only the term corresponding to the true class contributes to the loss.

### Average Loss

In a typical scenario where we have $N$ samples, the average categorical cross-entropy loss across all samples can be expressed as:

$$
L(y, \hat{y}) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{K} y_{n,i} \log(\hat{y}_{n,i})
$$

Where:
- $N$ is the number of samples.
- $y_{n,i}$ is the true label for sample $n$ and class $i$.
- $\hat{y}_{n,i}$ is the predicted probability for sample $n$ and class $i$.

## Numerical Stability

To avoid numerical issues (such as taking the logarithm of zero), it is common to clip the predicted probabilities:

$$
\hat{y}_{i} = \text{clip}(\hat{y}_{i}, \epsilon, 1 - \epsilon)
$$

Where $\epsilon$ is a small value (e.g., $1e-7$). This prevents undefined logarithmic values and ensures stable calculations.

## Conclusion

Categorical cross-entropy loss is a powerful and effective loss function for multi-class classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1. Understanding its mathematical foundation helps in optimizing and improving models in machine learning and deep learning applications.
