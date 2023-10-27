import numpy as np


class InputLayer:
    def __init__(self):
        self.next = None
        self.prev = None
        self.output = None

    def forward(self, inputs):
        self.output = inputs


class HiddenLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.next = None
        self.prev = None
        self.inputs = None
        self.output = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None
        self.weights = 0.01 * np.random.randn(num_of_inputs, num_of_neurons)  # Uniform distribution
        self.biases = np.zeros((1, num_of_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
