import math
import numpy as np

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

print('---'*10)
layer_outputs1 = np.array([[4.8, 1.21, 2.385],
                           [8.9, -1.41, 0.2],
                           [1.41, 1.051, 0.026]])
print(layer_outputs1)
print(f"Sum without axis: {np.sum(layer_outputs1)}")
print(f"This will be identical to the above since default is None: {np.sum(layer_outputs1, axis=None)}")
print(f"Another way to think of it w/ a matrix == axis 1': columns:\n {np.sum(layer_outputs1, axis=1, keepdims=True)}")

print("--"*20)
print("Sum the rows instead, like this w/ raw py")
for i in layer_outputs1:
    print(sum(i))

class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
