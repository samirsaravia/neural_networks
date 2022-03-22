import math
import numpy as np
from dense_layer_class import *
from nnfs.datasets import spiral_data
import nnfs
from relu_activation import *

nnfs.init()
# ---------------Softmax----------------
layer_outputs = [4.8, 1.21, 2.385]
# to exponentiate the outputs is used euler's number e=2.718271828486
E = math.e
exp_values = []

# Usual method
for output in layer_outputs:
    exp_values.append(E ** output)
print(f'Exponentiated values:{exp_values}')

# Now normalize values
norm_base = sum(exp_values)  # we sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print(f"Normalize exponentiated values: {norm_values}")
print(f"Sum of normalized values: {sum(norm_values)}")

# Numpy method
print("--------------------------" * 3)
exp_values_np = np.exp(layer_outputs)
print(f" Exponentiated values with numpy: {exp_values_np}")

# Normalize values
norm_values_np = exp_values_np / np.sum(exp_values_np)
print(f" Normalized values with numpy: {norm_values_np}")
print(f"Sum of normalized values: {np.sum(norm_values_np)}")

print('---' * 10)
layer_outputs1 = np.array([[4.8, 1.21, 2.385],
                           [8.9, -1.41, 0.2],
                           [1.41, 1.051, 0.026]])
print(layer_outputs1)
print(f"Sum without axis: {np.sum(layer_outputs1)}")
print(f"This will be identical to the above since default is None: {np.sum(layer_outputs1, axis=None)}")
print(f"Another way to think of it w/ a matrix == axis 1': columns:\n {np.sum(layer_outputs1, axis=1, keepdims=True)}")

print("--" * 20)
print("Sum the rows instead, like this w/ raw py")
for i in layer_outputs1:
    print(sum(i))


class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# create dataset
X, y = spiral_data(samples=100, classes=3)
# create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create Relu activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create softmax activation (to be used with Dense layer)
activation2 = Activation_Softmax()

# make a forward pass of our training data through this layer
dense1.forward(X)
# make a forward pass through activation func, it takes the output of first dense layer here
activation1.forward(dense1.output)

# Make a forward pass through second Dense layer it takes output of activation func of first layer as inputs
dense2.forward(activation1.output)

# Make a forward pass through activation function it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])
