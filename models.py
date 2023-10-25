# Module with neural network implementations
import matplotlib.pyplot as plt
import numpy as np
import pandas.core.frame


# noinspection PyPep8Naming,DuplicatedCode
class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers, classification_or_regression="C",
                 learning_rate=0.005, decay=5e-6):
        self.plot_counter = 1
        self.loss_plot_vals = []
        self.lr_plot_vals = []
        self.acc_plot_vals = []
        self.output = None
        self.input_layer = InputLayer()
        self.layers = []
        self.layers.append(HiddenLayer(input_dim, hidden_dim))
        self.layers.append(ActivationReLU())
        for _ in range(num_hidden_layers - 2):
            self.layers.append(HiddenLayer(hidden_dim, hidden_dim))
            self.layers.append(ActivationReLU())
        self.layers.append(HiddenLayer(hidden_dim, output_dim))
        self.optimizer = OptimizerAdam(learning_rate=learning_rate, decay=decay)
        if classification_or_regression == "C":
            self.loss = LossCategoricalCrossEntropy()
            self.layers.append(ActivationSoftmax())
            self.accuracy = AccuracyClassification()
        elif classification_or_regression == "R":
            self.loss = LossMeanSquaredError()
            self.layers.append(ActivationLinear())
            self.accuracy = AccuracyRegression()
        self.trainable_layers = []
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None, show_plots=False):
        self.accuracy.init(y)
        train_steps = 1
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        for epoch in range(epochs + 1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    start = step * batch_size
                    end = (step + 1) * batch_size
                    batch_X = X[start:end]
                    batch_y = y[start:end]
                output = self.forward(batch_X)
                loss = self.loss.calculate(output, batch_y)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
                self.backward(output, batch_y)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
                if show_plots:
                    self.lr_plot_vals.append([self.plot_counter, self.optimizer.current_learning_rate])
                    self.acc_plot_vals.append([self.plot_counter, accuracy])
                    self.loss_plot_vals.append([self.plot_counter, loss])
                    self.plot_counter += 1
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, '
                          f'acc: {accuracy:.3f}, '
                          f'loss: {loss:.3f}, '
                          f'lr: {self.optimizer.current_learning_rate}')
            epoch_loss = self.loss.calculate_accumulated()
            epoch_accuracy = self.accuracy.calculate_accumulated()
            print(f'training, '
                  f'acc: {epoch_accuracy:.3f}, '
                  f'loss: {epoch_loss:.3f}, '
                  f'lr: {self.optimizer.current_learning_rate}')
            if show_plots:
                self.loss_plot_vals.append([self.plot_counter, epoch_loss])
                self.acc_plot_vals.append([self.plot_counter, epoch_accuracy])
                self.lr_plot_vals.append([self.plot_counter, self.optimizer.current_learning_rate])
                self.plot_counter += 1
            if validation_data is not None:
                self.loss.new_pass()
                self.accuracy.new_pass()
                for step in range(validation_steps):
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                    else:
                        start = step * batch_size
                        end = (step + 1) * batch_size
                        batch_X = X_val[start:end]
                        batch_y = y_val[start:end]
                    output = self.forward(batch_X)
                    self.loss.calculate(output, batch_y)
                    predictions = self.output_layer_activation.predictions(output)
                    self.accuracy.calculate(predictions, batch_y)
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                print(f'validation, '
                      f'acc: {validation_accuracy:.3f}, '
                      f'loss: {validation_loss:.3f}')
        if show_plots:
            x_lr = [point[0] for point in self.lr_plot_vals]
            y_lr = [point[1] for point in self.lr_plot_vals]
            plt.plot(x_lr, y_lr, marker='o', linestyle='-')
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.title('Learning rate in each step')
            plt.show()
            x_acc = [point[0] for point in self.acc_plot_vals]
            y_acc = [point[1] for point in self.acc_plot_vals]
            plt.plot(x_acc, y_acc, marker='o', linestyle='-')
            plt.xlabel('Step')
            plt.ylabel('Accuracy')
            plt.title('Accuracy in each step')
            plt.show()
            x_loss = [point[0] for point in self.loss_plot_vals]
            y_loss = [point[1] for point in self.loss_plot_vals]
            plt.plot(x_loss, y_loss, marker='o', linestyle='-')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Loss in each step')
            plt.show()

    def validate(self, X, y):
        output = self.forward(X)
        loss = self.loss.calculate(output, y)
        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.calculate(predictions, y)
        print(
            f'validation, ' +
            f'acc {accuracy:.3f}, ' +
            f'loss: {loss:.3f}'
        )

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.output)
        return layer.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)


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


class Optimizer:
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: HiddenLayer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weights_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weights_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weights_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases
        layer.weights += weights_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1


class OptimizerAdam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: HiddenLayer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# noinspection PyTypeChecker
class Accuracy:
    def __init__(self):
        self.accumulated_count = None
        self.accumulated_sum = None

    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def compare(self, predictions, y):
        pass


class AccuracyRegression(Accuracy):
    def __init__(self):
        super().__init__()
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


class AccuracyClassification(Accuracy):
    def init(self, y):
        pass

    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
