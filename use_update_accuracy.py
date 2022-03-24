from full_code import *
import nnfs
from nnfs.datasets import vertical_data
from nnfs.datasets import spiral_data

nnfs.init()

# Create dataset
# X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)  # with vertical data

# create a model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_softmax()

# create loss function
loss_function = Loss_CategoricalCrossentropy()

# then create some variables to track the best loss and the associated weights and biases

# Helper variables
lowest_loss = 9999999  # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()

best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Generate a new set of weights for iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)

    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of the training data through this layer
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # If loss is smaller - print and save weights and  biases aside
    if loss < lowest_loss:
        print(f"New set of weights found, iteration: {iteration}/ loss:{loss} / accuracy:{accuracy}")
        print('-' * 50)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()

        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()

        lowest_loss = loss
    # revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
