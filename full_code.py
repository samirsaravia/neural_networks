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
    #Forward pass
    def forward(self, inputs):
        #Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize the for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
