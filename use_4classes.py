from full_code import *

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense Layer)
activation1 = Activation_ReLU()

# Create second Dense layer with 3 inpout features (as we take output of previous
# layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation ( to be used with dense layer)
activation2 = Activation_softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# ---------------perform-----------
# perform a forward pass of our training data through this layer
dense1.forward(X)

# perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# perform a forward pass through second dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# perform a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
print(activation2.output[:5])

# Perform a forward pass through loss function
# it takes the output of second dense layer ere and returns loss
loss = loss_function.calculate(activation2.output, y)
print(f"Loss: {loss}")
print(f"length of y: {len(y)}\nhow is like y: {y}")
