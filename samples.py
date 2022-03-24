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

print("--" * 30)
# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]
# Ground truth
target_output = [1, 0, 0]
loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])
print(loss)
print(-math.log(softmax_output[0]))
print(math.log(1.))
print(math.log(0.95))
# =================using euler's log=================
print("Euler's log")
b = 5.2
print(np.log(b))
print(math.e ** np.log(b))

softmax_outputs2 = [[0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]]
class_targets = [0, 1, 1]
for targ_idx, distributions in zip(class_targets, softmax_outputs2):
    print((distributions[targ_idx]))
soft_output = np.array([[0.7, 0.1, 0.2],
                        [0.1, 0.5, 0.4],
                        [0.02, 0.9, 0.08]])
neg_log = -np.log(soft_output[range(len(soft_output)), class_targets])
average_loss = np.mean(neg_log)
print(f"average loss: {average_loss}")  # mean = sum(iterable) / len (iterable)
class_targets2 = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 1, 0]])
print(f"length : {len(class_targets2.shape)}")
# Probabilities for target values only if categorical labels
if len(class_targets2.shape) == 1:
    correct_confidences = soft_output[range(len(soft_output)), class_targets2]
# mask values only for one-hot encoded labels
elif len(class_targets2.shape) == 2:
    correct_confidences = np.sum(soft_output * class_targets2, axis=1)
print(correct_confidences)
# print(-np.log(0))
# losses
neg_log = -np.log(correct_confidences)
avg_loss = np.mean(neg_log)
print(f"average loss: {avg_loss}")
print(-np.log(1e-7))
# -----------------------accuracy------------------
print("---" * 30)
print("--" * 15, "accuracy", "--" * 15)
# probabilities of 3 expamples
softmax_acc = np.array([[0.7, 0.2, 0.1],
                        [0.5, 0.1, 0.4],
                        [0.02, 0.9, 0.08]])

# Target (ground-truth) labels for 3 examples
class_acc_targets = np.array([0, 1, 1])

# Work out values along second axis
predictions = np.argmax(softmax_acc, axis=1)

# if targets are one-hot encoded - convert them
if len(class_acc_targets.shape) == 2:
    class_acc_targets = np.argmax(class_acc_targets, axis=1)
# True evaluates to 1; False to 0
accuracy = np.mean(predictions == class_acc_targets)
print(f"Acc: {accuracy}")
