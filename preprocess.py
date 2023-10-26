import numpy as np
import pickle
import os

from dataset import get_dataframes_heart_disease, get_dataframes_fashion_mnist, get_dataframes_forest_fires, \
    get_dataframes_wine_quality
from sklearn.model_selection import train_test_split


# noinspection PyPep8Naming
def get_train_and_test_datasets_heart_disease():
    X, y = get_dataframes_heart_disease()
    X = X.drop(['ca', 'thal'], axis=1)
    X_train, X_test, y_trainP, y_testP = train_test_split(X, y, test_size=0.15, shuffle=True)
    y_train = y_trainP.T.to_numpy()[0]
    y_test = y_testP.T.to_numpy()[0]
    return X_train, X_test, y_train, y_test


# noinspection PyPep8Naming
def get_train_and_test_datasets_wine_quality():
    X, y = get_dataframes_wine_quality()
    X_normalised = (2 * (X - X.min()) / (X.max() - X.min())) - 1
    y_normalised = y / 10
    X_train, X_test, y_trainP, y_testP = train_test_split(X_normalised, y_normalised, test_size=0.2, shuffle=True)
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_trainP.to_numpy()
    y_test = y_testP.to_numpy()
    return X_train, X_test, y_train, y_test


get_train_and_test_datasets_wine_quality()


# noinspection PyPep8Naming
def get_train_and_test_datasets_fashion_mnist():
    data_folder = 'data/processed/fashion_mnist_datasets'
    X_filename = os.path.join(data_folder, 'X.pkl')
    y_filename = os.path.join(data_folder, 'y.pkl')
    X_test_filename = os.path.join(data_folder, 'X_test.pkl')
    y_test_filename = os.path.join(data_folder, 'y_test.pkl')

    if os.path.exists(X_filename) and os.path.exists(y_filename) and os.path.exists(X_test_filename) and os.path.exists(
            y_test_filename):
        # If the serialized files exist, load the data from them
        with open(X_filename, 'rb') as file:
            X = pickle.load(file)
        with open(y_filename, 'rb') as file:
            y = pickle.load(file)
        with open(X_test_filename, 'rb') as file:
            X_test = pickle.load(file)
        with open(y_test_filename, 'rb') as file:
            y_test = pickle.load(file)
    else:
        # If the serialized files don't exist, generate the data
        X, y, X_test, y_test = get_dataframes_fashion_mnist()
        keys = np.array(range(X.shape[0]))
        np.random.shuffle(keys)
        X = X[keys]
        y = y[keys]
        X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
        X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

        # Save the data to serialized files
        with open(X_filename, 'wb') as file:
            pickle.dump(X, file)
        with open(y_filename, 'wb') as file:
            pickle.dump(y, file)
        with open(X_test_filename, 'wb') as file:
            pickle.dump(X_test, file)
        with open(y_test_filename, 'wb') as file:
            pickle.dump(y_test, file)

    return X, X_test, y, y_test


# noinspection PyPep8Naming
def get_preprocessed_train_and_test_datasets_forest_fires():
    df = get_dataframes_forest_fires()
    # Logarithm transform because of the skew towards 0.0
    df['area'] = np.log1p(df['area'])
    # Drop the first row (labels)
    df = df.iloc[1:]

    # Map month and day to integers
    month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    day_mapping = {'sun': 1, 'mon': 2, 'tue': 3, 'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}

    df['month'] = df['month'].map(month_mapping)
    df['day'] = df['day'].map(day_mapping)

    # Convert to int
    df = df.astype({'month': int, 'day': int})

    # Splitting the DataFrame into features (X) and target (y)
    X = df.drop(columns=['area'])  # All columns except 'area'
    y = df['area']  # 'area' column as the target variable

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test
