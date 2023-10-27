import numpy as np


class ActivationFunction:
    def __init__(self):
        self.next = None
        self.prev = None
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        pass

    def backward(self, dvalues):
        pass

    @staticmethod
    def predictions(outputs):
        pass


class ActivationReLU(ActivationFunction):  # Other gradient and derivatives calculations than sigmoid
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    @staticmethod
    def predictions(outputs):
        return outputs


class ActivationSoftmax(ActivationFunction):  # Depends on what we want to achieve - good for 1 of k classes
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalue)

    @staticmethod
    def predictions(outputs):
        return np.argmax(outputs, axis=1)


class ActivationLinear(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    @staticmethod
    def predictions(outputs):
        return outputs
