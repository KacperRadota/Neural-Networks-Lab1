# Module with neural network implementations
import preprocess
import numpy as np
import pandas.core.frame


class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        self.output = None
        self.loss = None
        self.accuracy = None
        self.layers = []
        self.layers.append(HiddenLayer(input_dim, hidden_dim))
        for _ in range(num_hidden_layers):
            self.layers.append(HiddenLayer(hidden_dim, hidden_dim))
        self.layers.append(HiddenLayer(hidden_dim, output_dim))
        self.activations = []
        self.activations.append(-1)  # First index as input layer so no activation function
        for _ in range(len(self.layers) - 2):
            self.activations.append(ActivationReLU())
        self.activations.append(ActivationSoftmaxXLossCategoricalCrossEntropy())
        self.optimizer = Optimizer()

    def forward(self, inputs, y_true):
        prev_output = None
        for i, layer in enumerate(self.layers):
            if i == (len(self.layers) - 1):  # Softmax and loss function case
                self.loss = self.activations[i].forward(prev_output, y_true)
                predictions = np.argmax(self.activations[i].output, axis=1)
                if len(y_true.shape) == 2:
                    y_true = np.argmax(y_true, axis=1)
                self.accuracy = np.mean(predictions == y_true)
                break
            if i == 0:  # Input layer case
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
        self.dinputs = np.empty_like(dvalues)
        for i, (single_output, single_dvalue) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalue)


class Loss:
    def __init__(self):
        self.dinputs = None

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, y_pred, y_true) -> pandas.core.frame.DataFrame:
        pass

    def backward(self, dvalues, y_true):
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

    def backward(self, dvalues, y_true):
        # When labels are sparse -> one-hot vector
        if len(y_true.shape) == 1:
            labels = len(dvalues[0])
            y_true = np.eye(labels)[y_true]
        samples = len(dvalues)
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


# Created like that for faster computations than softmax and loss separately
class ActivationSoftmaxXLossCategoricalCrossEntropy:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, y_true) -> pandas.core.frame.DataFrame:
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        # If labels are one-hot -> discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, layer: HiddenLayer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


# noinspection PyPep8Naming
def run():
    X, y = preprocess.get_preprocessed_datasets()
    nn = NeuralNetwork(11, 10, 5, 2)
    nn.forward(X, y)
    print(nn.output)


if __name__ == "__main__":
    run()
