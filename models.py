# Module with neural network implementations
import preprocess
import numpy as np
import pandas.core.frame


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        self.output = None
        self.layers = []
        self.layers.append(HiddenLayer(input_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            self.layers.append(HiddenLayer(hidden_dim, hidden_dim))
        self.layers.append(HiddenLayer(hidden_dim, output_dim))
        self.activations = []
        self.activations.append(-1)  # First index as input layer so no activation function
        for _ in range(len(self.layers) - 2):
            self.activations.append(ActivationReLU())
        self.activations.append(ActivationSoftmax())

    def forward(self, inputs):
        prev_output = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(inputs)
                prev_output = layer.output
                continue
            layer.forward(prev_output)
            self.activations[i].forward(layer.output)
            prev_output = self.activations[i].output
        self.output = prev_output

    def backward(self):
        pass


class InputLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.output = None
        self.num_of_inputs = num_of_inputs
        self.num_of_neurons = num_of_neurons

    def forward(self, inputs):
        self.output = inputs


class HiddenLayer:
    def __init__(self, num_of_inputs, num_of_neurons):
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


class ActivationFunction:
    def __init__(self):
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        pass

    def backward(self, dvalues):
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


class ActivationSoftmax(ActivationFunction):  # Depends on what we want to achieve - good for 1 of k classes
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        self.inputs = inputs
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exponential_values / np.sum(exponential_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        pass


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true) -> pandas.core.frame.DataFrame:
        pass


class LossCategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true) -> pandas.core.frame.DataFrame:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidences)


# noinspection PyPep8Naming
def run():
    X, y = preprocess.get_preprocessed_datasets()
    nn = NeuralNetwork(11, 10, 5, 2)
    nn.forward(X)
    print(nn.output)


if __name__ == "__main__":
    run()
