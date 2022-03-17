import nnfs
import numpy as np

nnfs.init()

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # will return n_inputs number rows and n_neurons number columns
biases = np.zeros((1, n_neurons))  # will return 1D , one row and n_neurons number columns

print(weights)
print(biases)
