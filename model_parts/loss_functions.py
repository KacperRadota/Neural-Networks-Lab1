import numpy as np
import pandas.core.frame

from model_parts.activation_functions import ActivationSoftmax


class Loss:
    def __init__(self):
        self.accumulated_count = None
        self.accumulated_sum = None
        self.trainable_layers = None
        self.dinputs = None

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        return data_loss

    def calculate_accumulated(self):
        data_loss = self.accumulated_sum / self.accumulated_count
        return data_loss

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

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


class LossMeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true) -> pandas.core.frame.DataFrame:
        sample_loses = np.mean((y_true - y_pred) ** 2, axis=1)
        return sample_loses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


# Created like that for faster computations than softmax and loss separately
class ActivationSoftmaxXLossCategoricalCrossEntropy:
    def __init__(self):
        self.next = None
        self.prev = None
        self.dinputs = None
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    def forward(self, inputs, y_true):
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
