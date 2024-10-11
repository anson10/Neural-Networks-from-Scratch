### Linear Activation Function

For a given input $x$, the linear activation function is:

$$
f(x) = x
$$

This means that the output is exactly the same as the input.

### Mean Squared Error (MSE)

Mean Squared Error (MSE) is a common loss function used in regression tasks. It measures the average of the squares of the errors, i.e., the average squared difference between the predicted values and the actual values.

#### Formula

For a dataset with $n$ examples, the mean squared error is defined as:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$ is the actual value (ground truth).
- $\hat{y}_i$ is the predicted value from the model.
- $n$ is the total number of examples.

#### Characteristics
- **Sensitive to Outliers**: Since it squares the errors, larger errors have a disproportionately high effect on the MSE.
- **Non-negative**: MSE is always non-negative, as it squares the differences.
- **Units**: The units of MSE are the square of the units of the output variable, which can sometimes make interpretation difficult.

### Derivative of MSE

To minimize the MSE during the optimization process, we often need to calculate its derivative with respect to the predicted values. The derivative of the MSE is given by:

$$
\frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n} (\hat{y}_i - y_i)
$$

This derivative indicates how the loss changes with respect to changes in the predicted values. It is used in optimization algorithms like gradient descent to update the model parameters.


### Mean Absolute Error (MAE)

Mean Absolute Error (MAE) is another common loss function used in regression tasks. It measures the average of the absolute differences between the predicted values and the actual values.

#### Formula

For a dataset with $n$ examples, the mean absolute error is defined as:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

Where:
- $y_i$ is the actual value (ground truth).
- $\hat{y}_i$ is the predicted value from the model.
- $n$ is the total number of examples.

#### Characteristics
- **Robust to Outliers**: MAE is less sensitive to outliers compared to MSE, as it uses absolute values instead of squaring the differences.
- **Non-negative**: MAE is always non-negative, as it measures absolute differences.
- **Units**: The units of MAE are the same as the units of the output variable, making it easier to interpret.

### Derivative of MAE

To minimize the MAE during optimization, we often need to calculate its derivative with respect to the predicted values. The derivative of the MAE is given by:

$$
\frac{\partial \text{MAE}}{\partial \hat{y}_i} = 
\begin{cases} 
1 & \text{if } \hat{y}_i > y_i \\
-1 & \text{if } \hat{y}_i < y_i \\
0 & \text{if } \hat{y}_i = y_i 
\end{cases}
$$

This derivative indicates the direction in which to adjust the predicted values to reduce the error. In optimization algorithms like gradient descent, it guides the update of model parameters based on whether the prediction is above or below the actual value.


### Accuracy in Regression

In regression tasks, traditional accuracy is not directly applicable as it is in classification tasks. Instead, we assess regression models using various performance metrics that quantify the differences between predicted and actual values. However, we can discuss related concepts, including precision, accuracy, and their relevance in a regression context.

#### Key Metrics for Regression

1. **Mean Squared Error (MSE)**:
   - Measures the average of the squares of the errors—that is, the average squared difference between predicted and actual values.
   - **Formula**:

   $$
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$

2. **Mean Absolute Error (MAE)**:
   - Measures the average of the absolute differences between predicted and actual values.
   - **Formula**:

   $$
   \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   $$

3. **Root Mean Squared Error (RMSE)**:
   - The square root of the average of squared differences, providing a measure of error in the same units as the target variable.
   - **Formula**:

   $$
   \text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
   $$

4. **R-squared (R²)**:
   - Indicates the proportion of variance in the dependent variable explained by the independent variables.
   - **Formula**:

   $$
   R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
   $$

   Where:
   - $\text{SS}_{\text{res}}$ is the residual sum of squares.
   - $\text{SS}_{\text{tot}}$ is the total sum of squares.

### Conceptualizing Accuracy in Regression

While accuracy is not traditionally defined for regression, we can understand the model's performance through several perspectives:

#### 1. **Threshold-based Accuracy**

In cases where we categorize continuous outputs, we can define accuracy using a threshold. This is particularly useful when dealing with binary outcomes:

- **Example**: In predicting whether a house price is "Affordable" or "Expensive," we can set a price threshold (e.g., $300,000). If the model predicts correctly whether a price is above or below this threshold, we can calculate accuracy.

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}
$$

#### 2. **Precision and Recall in Regression Context**

While precision and recall are primarily classification metrics, they can be adapted to evaluate the performance of regression models when they are converted into binary outcomes:

- **Precision** measures the accuracy of the positive predictions:

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

- **Recall** (or Sensitivity) measures the ability to find all the relevant cases:

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

#### 3. **F1 Score**

The F1 Score is the harmonic mean of precision and recall, providing a balance between the two. It can be useful when we need a single metric to evaluate performance in a thresholded regression task:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Conclusion

In regression analysis, while traditional accuracy may not apply, several metrics such as MSE, MAE, RMSE, and R-squared serve as proxies for assessing model performance. By conceptualizing accuracy through thresholding and using adapted metrics like precision, recall, and the F1 score, we can gain insights into how well our regression models perform and make informed decisions for improvements.
