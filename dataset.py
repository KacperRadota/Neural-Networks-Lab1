# Dataset implementation
from ucimlrepo import fetch_ucirepo
from zipfile import ZipFile
import os
import urllib
import urllib.request
import numpy as np
import cv2
import pandas as pd


# noinspection PyPep8Naming
def get_dataframes_heart_disease():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return X, y


# noinspection PyPep8Naming
def get_dataframes_wine_quality():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets
    print(X)
    return X, y


def get_dataframes_fashion_mnist():
    return create_data_fashion_mnist('data/raw/fashion_mnist_images')


def download_and_prepare_fashion_mnist_file():
    url = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    file = 'data/raw/fashion_mnist_images.zip'
    folder = 'data/raw/fashion_mnist_images'
    if not os.path.isfile(file):
        print(f'Downloading {url} and saving as {file}...')
    urllib.request.urlretrieve(url, file)
    print('Unzipping images...')
    with ZipFile(file) as zip_images:
        zip_images.extractall(folder)
    print('Done!')


# noinspection PyPep8Naming
def load_fashion_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
    return np.array(X), np.array(y).astype('uint8')


# noinspection PyPep8Naming
def create_data_fashion_mnist(path):
    X, y = load_fashion_mnist_dataset('train', path)
    X_test, y_test = load_fashion_mnist_dataset('test', path)
    return X, y, X_test, y_test


def get_dataframes_forest_fires():
    df = pd.read_csv("data/raw/forest+fires/forestfires.csv")
    return df
