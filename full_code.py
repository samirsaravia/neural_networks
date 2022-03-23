import nnfs
import numpy as np
from nnfs.datasets import spiral_data

# nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weighs and biases
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)


class Activation_softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Common loss class
class Loss:
    # Calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):
        # Calculate samples losses
        sample_losses = self.forward(output, y)
        # calculate mean loss
        data_loss = np.mean(sample_losses)
        # return loss
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        # losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
