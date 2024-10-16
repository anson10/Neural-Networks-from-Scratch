# Prediction / Inference in Deep Learning

Prediction (or Inference) is the phase where a trained deep learning model is used to make predictions on unseen data. This stage comes after the model has been trained on a dataset. Inference is critical for applying the model to real-world tasks.

## Steps in Prediction / Inference

1. **Input Data**:  
   During inference, you provide new data (input) that the model hasn't seen during training.
   
2. **Forward Propagation**:  
   The input data is passed through the neural network in a forward direction. The weights and biases learned during training are used to calculate the output (predictions).

3. **No Gradients**:  
   Unlike training, the backward pass (where gradients are calculated) is not performed during inference. Therefore, gradient calculations are usually turned off to save computation time.

4. **Output Predictions**:  
   The final layer of the model provides predictions. Depending on the task, this could be:
   - **Classification**: Producing class probabilities or labels.
   - **Regression**: Producing a continuous output (like predicting a number).
   
5. **Post-processing**:  
   Sometimes, additional processing is required on the model's output. For example, converting probabilities into class labels (e.g., selecting the class with the highest probability).

6. **Evaluation (Optional)**:  
   If the ground truth (correct labels) is available, the model's predictions can be evaluated using accuracy, loss, or other metrics.

## Example: Classification Task

1. **Input**: A new image (e.g., 28x28 pixel image).
2. **Forward Pass**: The image is passed through the model layers.
3. **Output**: The model outputs probabilities for each class.
4. **Prediction**: The class with the highest probability is chosen as the predicted label.

## Considerations for Inference

- **Batching**: Inputs are often processed in batches during inference to speed up the computation.
- **Memory Usage**: Inference generally consumes less memory compared to training since it doesn’t require storing gradients.
- **Optimizations**: Various techniques (e.g., quantization, model pruning) can be applied to optimize models for faster inference.

## Summary

- Inference is about using a trained model to make predictions.
- It involves forward propagation, without gradient calculation.
- The output is either class predictions (for classification) or continuous values (for regression).

