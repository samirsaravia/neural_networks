# Single neuron

# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2
# output = (inputs[0] * weights[0] +
#           inputs[1] * weights[1] +
#           inputs[2] * weights[2] +
#           inputs[3] * weights[3] + bias)


# layer of neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# for each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each inpout and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input * weight
    # Add bias
    neuron_output += neuron_bias
    # put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

# Dot product
# we need to multiply our weight and inputs of the same index values
a = [1, 2, 3]
b = [2, 3, 4]

dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
print(dot_product)
