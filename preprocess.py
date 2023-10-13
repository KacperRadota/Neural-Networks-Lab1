from dataset import get_dataframes


# noinspection PyPep8Naming
def get_preprocessed_datasets():
    X, y = get_dataframes()
    return X.drop(['ca', 'thal'], axis=1), y
