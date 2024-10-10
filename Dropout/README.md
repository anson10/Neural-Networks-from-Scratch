# Dropout in Neural Networks

Dropout is a regularization technique used in neural networks to prevent overfitting. During the training process, it temporarily removes (or "drops out") a random subset of neurons in a layer. This forces the network to develop redundancy, making it more robust and less likely to rely on particular neurons, which helps improve generalization.

## Forward Pass with Dropout

During the forward pass, we apply a dropout mask to the layer's output. This mask is an array with the same shape as the layer's output, but it is filled with values drawn from a **Bernoulli distribution**. A Bernoulli distribution is a discrete probability distribution where we get a value of 1 with a probability of $p$, and a value of 0 with a probability of $q$, where $q = 1 - p$.

For a random value $r_i$ drawn from this distribution:

$P(r_i = 1) = p$

$P(r_i = 0) = q = 1 - p$

This means the probability of the value being 1 is $p$, and the probability of it being 0 is $q$. We can express this as:

$r_i \sim Bernoulli(p)$

In this context, $r_i$ is a value from the Bernoulli distribution, which is used as a mask for the layer's output. During the forward pass, we apply this mask to filter the neurons. Neurons that correspond to the 1's in the mask will remain active, while those that correspond to the 0's will be dropped (temporarily deactivated).

### Why Use Dropout?

- **Reduce Overfitting**: Dropout forces the network to learn robust features by randomly deactivating neurons. This prevents the model from becoming too dependent on specific neurons and helps it generalize better to new, unseen data.
- **Increases Model Complexity**: By introducing randomness, dropout essentially trains different "sub-networks" within the model, which adds diversity to the learned features.

## Explanation of `np.random.binomial()` and Dropout in Neural Networks

### 1. What is `np.random.binomial()`?
The function `np.random.binomial()` simulates multiple trials of a binomial distribution, like flipping a coin.

- **n**: Number of trials or "experiments" per test.
- **p**: Probability of success (e.g., how likely it is that you'll get a `1` in a coin toss).
- **size**: How many independent tests (arrays) to run.

For example:
```python
np.random.binomial(2, 0.5, size=10)
```
This simulates flipping 2 coins (with a 50% chance of getting heads) 10 times, giving an array of results. Each result is the total number of heads (successes) in 2 coin tosses.

Result could be something like:
```python
array([0, 0, 1, 2, 0, 2, 0, 1, 0, 2])
```
This represents the outcomes of 10 tests, where each element sums the result of 2 coin tosses
```python
array([0, 1, 1, 1, 1])
```

### 2. Using np.random.binomial() for Dropout
In dropout, we randomly turn off neurons (i.e., set their output to zero) during training to prevent overfitting.

- *dropout_rate* is the fraction of neurons we want to drop (set to 0).
For example, if we want to drop 20% of the neurons:

```python
dropout_rate = 0.2
np.random.binomial(1, 1 - dropout_rate, size=5)
```
This generates an array of 1s and 0s. A value of `1` means the neuron is kept, and `0` means the neuron is dropped. The probability of getting a `1` (keeping the neuron) is `1 - dropout_rate`, and the probability of getting a `0` (dropping the neuron) is `dropout_rate`.

For example:
```python
array([0, 1, 1, 1, 1])
```
Here, the first neuron is dropped, and the rest are kept.

### 3. Applying Dropout to Neural Network Output

Let's say we have the following neural network output:

```python
example_output = np.array([ 0.27, -1.03, 0.67, 0.99, 0.05, -0.37, -2.01, 1.13, -0.07, 0.73])
```

To apply dropout with a 30% dropout rate (i.e., 30% of neurons should be dropped), we use:

```python
dropout_rate = 0.3
example_output *= np.random.binomial(1, 1 - dropout_rate, example_output.shape)
```

This multiplies each element of `example_output` by either `0` (drop) or `1` (keep), based on the binomial distribution. The resulting `example_output` might look like:

```list
[ 0.27, -1.03, 0.00, 0.99, 0.00, -0.37, -2.01, 1.13, -0.07, 0.00 ]
```
Here, the third, fifth, and last neurons have been dropped (set to `0`).

### 4. Scaling during Training

When you apply dropout, you're dropping some neurons' outputs, but this means the inputs to the next layer during training will be smaller than during prediction (since some neurons are randomly zeroed out).

To correct this, you scale the outputs during training by dividing by (1 - dropout_rate), which compensates for the dropped neurons. This keeps the overall magnitude of the output stable.

Example with scaling:
```python
example_output *= np.random.binomial(1, 1 - dropout_rate, example_output.shape) / (1 - dropout_rate)
```

This scaling ensures that the magnitude of the neuron outputs remains similar between training and prediction, despite the random dropping of neurons during training.

### 5. Dropout Example with Mean Sum Validation
In the final example, a test is done to check that the sum of the outputs (after scaling) converges to a similar value as the original sum, despite dropout.
```python
dropout_rate = 0.2
example_output = np.array([0.27, -1.03, 0.67, 0.99, 0.05, -0.37, -2.01, 1.13, -0.07, 0.73])
print(f'sum initial {sum(example_output)}')  # Original sum

```
The code then applies dropout 10,000 times and calculates the mean sum of the scaled outputs:

```python
sums = []
for i in range(10000):
    example_output2 = example_output * np.random.binomial(1, 
                    1-dropout_rate, example_output.shape) / (1 - dropout_rate)
    sums.append(sum(example_output2))

print(f'mean sum: {np.mean(sums)}')
```

This shows that, after many trials, the mean sum of the scaled outputs closely approximates the original sum, proving that the scaling ensures consistent behavior during training and prediction.


