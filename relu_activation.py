from nnfs.datasets import spiral_data
from dense_layer_class import *


class Activation_ReLU:
    # forward pass
    def forward(self, inputs):
        # calculate output values from input
        self.output = np.maximum(0, inputs)


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func
# Takes in output from previous layer
activation1.forward(dense1.output)

print(activation1.output[:5])
