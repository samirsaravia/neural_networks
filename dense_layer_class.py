import numpy as np
# from nnfs.datasets import spiral_data
# import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # initialize weights and biases
        self.weights = 0.01 * np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weighs and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# create a dataset
# x, y = spiral_data(samples=100, classes=3)
#
# # Create dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense(2, 3)
#
# # Perform a forward pass of our training data through this layer
# dense1.forward(x)
#
# #  Let's see output of the first few samples
# # print(dense1.output[:])
