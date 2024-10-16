# Model Evaluation in Deep Learning

Model evaluation is critical to assess how well a deep learning model generalizes to new data. Here are the key aspects:

## 1. **Evaluation Metrics**

- **Accuracy**: Proportion of correctly predicted samples.
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to identify all positive samples.
- **F1 Score**: Balance between precision and recall.
- **Loss**: Measures the model’s error during training (e.g., Cross-Entropy, MSE).
- **ROC-AUC**: Evaluates performance at different thresholds, ideal for imbalanced datasets.

## 2. **Validation Methods**

- **Holdout**: Splitting data into training, validation, and test sets.
- **Cross-Validation**: Repeatedly training on different subsets of data.

## 3. **Confusion Matrix**

A breakdown of predicted vs. actual outcomes:
- **True Positives (TP)**
- **False Positives (FP)**
- **True Negatives (TN)**
- **False Negatives (FN)**

## 4. **Overfitting and Underfitting**

- **Overfitting**: High accuracy on training but poor on unseen data.
- **Underfitting**: Poor performance on both training and test data.

## 5. **Visualizations**

- **Learning Curves**: Plotting accuracy/loss over epochs to detect overfitting.
- **Confusion Matrix Heatmap**: Visual representation of the matrix.
- **ROC Curve**: Plots True Positive Rate vs. False Positive Rate.
