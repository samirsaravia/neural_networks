import numpy as np

#
# a = [1, 2, 3]
# b = [2, 3, 4]
#
# dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

# inputs = [1.0, 2.0, 3.0, 2.5]
# weights = [0.2, 0.8, - 0.5, 1.0]
# bias = 2.0
#
# outputs = np.dot(weights, inputs) + bias

# layer of neurons
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, - 0.5, 1],
           [0.5, - 0.91, 0.26, - 0.5],
           [- 0.26, - 0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

a = [1, 2, 3]
print(np.expand_dims(np.array(a), axis=1))  # axis so pode ser 1(vertical) ou 0 (horizontal)

