# Training loop
import numpy as np

from models import NeuralNetwork
from preprocess import get_train_and_test_datasets_heart_disease, get_train_and_test_datasets_fashion_mnist, \
    get_preprocessed_train_and_test_datasets_forest_fires, get_train_and_test_datasets_wine_quality


# noinspection PyPep8Naming
def train_heart_disease(batch_size=40):
    X_train, X_test, y_train, y_test = get_train_and_test_datasets_heart_disease()
    nn = NeuralNetwork(11, 30, 5, 2, "C")
    nn.train(X_train, y_train, epochs=1000, batch_size=batch_size, print_every=100, validation_data=(X_test, y_test))


# noinspection PyPep8Naming
def train_wine_quality():
    X_train, X_test, y_train, y_test = get_train_and_test_datasets_wine_quality()
    nn = NeuralNetwork(11, 32, 1, 2, "R", decay=0, learning_rate=0.005, accuracy_precision=0.1)
    nn.train(X_train, y_train, epochs=1000, batch_size=1024, print_every=1, validation_data=(X_test, y_test))


# noinspection PyPep8Naming
def train_fashion_mnist():
    X_train, X_test, y_train, y_test = get_train_and_test_datasets_fashion_mnist()
    nn = NeuralNetwork(784, 64, 10, 2, "C", decay=5e-5)
    nn.train(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, print_every=100,
             show_plots=True)


# noinspection PyPep8Naming
def train_forest_fires():
    X_train, X_test, y_train, y_test = get_preprocessed_train_and_test_datasets_forest_fires()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()[:, np.newaxis]
    y_test = y_test.to_numpy()[:, np.newaxis]
    nn = NeuralNetwork(12, 64, 1, 2, "R", decay=1e-7, learning_rate=0.1)
    nn.train(X_train, y_train, validation_data=(X_test, y_test), epochs=20000, batch_size=32, print_every=500)


# train_heart_disease()
# train_fashion_mnist()
# train_forest_fires()
train_wine_quality()
