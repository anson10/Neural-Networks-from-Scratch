# Softmax Activation Derivative

The next calculation that we need to perform is the partial derivative of the Softmax function, which is a bit more complicated task than the derivative of the Categorical Cross-Entropy loss. Let's remind ourselves of the equation of the Softmax activation function and define the derivative:

$$
S_{i,j} = \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}} \rightarrow \frac{\partial S_{i,j}}{\partial z_{i,k}} = ?
$$

Where:
- $S_{i,j}$ denotes the j-th Softmax output of the i-th sample,
- $z_{i,j}$ is the j-th Softmax input of the i-th sample,
- $L$ is the number of inputs,
- $z_{t,k}$ is the k-th Softmax input of the i-th sample.

As we described in chapter 4, the Softmax function equals the exponentiated input divided by the sum of all exponentiated inputs. In other words, we need to exponentiate all of the values first, then divide each of them by the sum of all of them to perform the normalization. Each input to the Softmax impacts each of the outputs, and we need to calculate the partial derivative of each output with respect to each input. From the programming side of things, if we calculate the impact of one list on the other list, we'll receive a matrix of values as a result. That's exactly what we'll calculate here — we'll calculate the Jacobian matrix (which we'll explain later) of the vectors, which we'll dive deeper into soon.

To calculate this derivative, we need to first define the derivative of the division operation:

$$
f(x) = \frac{q(x)}{h(x)} \rightarrow f'(x) = \frac{q'(x) \cdot h(x) - g(x) \cdot h'(x)}{[h(x)]^2}
$$

In order to calculate the derivative of the division operation, we need to take the derivative of the numerator multiplied by the denominator, subtract the numerator multiplied by the derivative of the denominator from it, and then divide the result by the squared denominator.

We can now start solving the derivative:

$$
\frac{\partial S_{i,j}}{\partial z_{i,k}} = \frac{\partial \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}}}{\partial z_{i,k}} =
$$

Let's apply the derivative of the division operation:

$$
= \frac{\frac{\partial}{\partial z_{i,k}} e^{z_{i,j}} \cdot \sum_{l=1}^{L} e^{z_{i,l}} - e^{z_{i,j}} \cdot \frac{\partial}{\partial z_{i,k}} \sum_{l=1}^{L} e^{z_{i,l}}}{\left[ \sum_{l=1}^{L} e^{z_{i,l}} \right]^2}
$$

At this step, we have two partial derivatives present in the equation. For the one on the right side of the numerator (right side of the subtraction operator):

$$
\frac{\partial}{\partial z_{i,k}} \sum_{l=1}^{L} e^{z_{i,l}}
$$

We need to calculate the derivative of the sum of the constant \( e \) (Euler's number), raised to the power of \( z_{i,l} \) (where \( l \) denotes consecutive indices from 1 to the number of the Softmax outputs \( L \)) with respect to \( z_{i,k} \). The derivative of the sum operation is the sum of derivatives, and the derivative of the constant \( e \) raised to power \( n \) (i.e., \( e^n \)) with respect to \( n \) equals \( e^n \):

$$
\frac{d}{dn} e^n = e^n \cdot \frac{d}{dn} n = e^n \cdot 1 = e^n
$$

It is a special case when the derivative of an exponential function equals this exponential function itself, as its exponent is exactly what we are deriving with respect to, thus its derivative equals 1. We also know that the range 1...L contains \( k \) (where \( k \) is one of the indices from this range) exactly once and then, in this case, the derivative is going to equal \( e^{z_{i,k}} \) (as \( j \) equals \( k \)) and \( 0 \) otherwise (when \( j \) does not equal \( k \) as \( z_{i,l} \) won't contain \( z_{i,k} \) and will be treated as a constant — the derivative of the constant equals 0):

$$
\frac{\partial}{\partial z_{i,k}} \sum_{l=1}^{L} e^{z_{i,l}} = 0 + 0 + \cdots + e^{z_{i,k}} + \cdots + 0 + 0 = e^{z_{i,k}}
$$

The derivative on the left side of the subtraction operator in the denominator is a slightly different case:

$$
\frac{\partial}{\partial z_{i,k}} e^{z_{i,j}}
$$

It does not contain the sum over all of the elements like the derivative we solved moments ago, so it can become either 0 if \( j \neq k \) or \( e^{z_{i,j}} \) if \( j = k \). That means, starting from this step, we need to calculate the derivatives separately for both cases. Let's start with \( j = k \).

In the case of \( j = k \), the derivative on the left side is going to equal \( e^{z_{i,j}} \) and the derivative on the right solves to the same value in both cases. Let's substitute them:

$$
= \frac{e^{z_{i,j}} \cdot \sum_{l=1}^{L} e^{z_{i,l}} - e^{z_{i,j}} \cdot e^{z_{i,j}}}{\left[ \sum_{l=1}^{L} e^{z_{i,l}} \right]^2}
$$

The numerator contains the constant \( e^{z_{i,j}} \) in both the minuend (the value we are subtracting from) and subtrahend (the value we are subtracting from the minuend) of the subtraction operation. Because of this, we can regroup the numerator to contain this value multiplied by the subtraction of their current multipliers. We can also write the denominator as a multiplication of the value instead of using the power of 2:

$$
= \frac{e^{z_{i,j}} \cdot \left( \sum_{l=1}^{L} e^{z_{i,l}} - e^{z_{i,k}} \right)}{\sum_{l=1}^{L} e^{z_{i,l}} \cdot \sum_{l=1}^{L} e^{z_{i,l}}}
$$

Then let's split the whole equation into 2 parts:

$$
= \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}} \cdot \frac{\sum_{l=1}^{L} e^{z_{i,l}} - e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}}
$$

We moved \( e^{z_{i,j}} \) from the numerator and the sum from the denominator to its own fraction, and the content of the parentheses in the numerator, and the other sum from the denominator as another fraction, both joined by the multiplication operation. Now we can further split the "right" fraction into two separate fractions:

$$
= \frac{e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}} \cdot \left( \frac{\sum_{l=1}^{L} e^{z_{i,j}}}{\sum_{l=1}^{L} e^{z_{i,l}}} - \frac{e^{z_{i,k}}}{\sum_{l=1}^{L} e^{z_{i,l}}} \right)
$$

In this case, as it's a subtraction operation, we separated both values from the numerator, dividing them both by the denominator and applying the subtraction operation between new fractions. If we look closely, the “left" fraction turns into the Softmax function's equation, as well as the "right” one, with the middle fraction solving to 1 as the numerator and the denominator are the same values:

$$
= S_{i,j} \cdot (1 - S_{i,k})
$$

Note that the “left" Softmax function carries the parameter \( j \), and the “right" one — both came from their numerators, respectively.

## Full solution:

$$
\frac{\partial S_{i,j}}{\partial z_{i,k}} = \frac{e^{z_{i,j}} \cdot \sum_{l=1}^{L} e^{z_{i,l}} - e^{z_{i,j}} \cdot e^{z_{i,k}}}{\left[ \sum_{l=1}^{L} e^{z_{i,l}} \right]^2} =
\begin{cases}
S_{i,j} \cdot (1 - S_{i,k}), & \text{if } j = k \\
-S_{i,j} \cdot S_{i,k}, & \text{if } j \neq k
\end{cases}
$$

The derivative is equal to \( S_{i,j} \cdot (1 - S_{i,j}) \) if the index is the same (which means it will also equal the partial derivative of the Softmax function with respect to itself). In case the indices differ, it results in \( - S_{i,j} \cdot S_{i,k} \) (the product of both outputs).
