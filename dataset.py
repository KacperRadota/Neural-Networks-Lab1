# Dataset implementation
from ucimlrepo import fetch_ucirepo


# noinspection PyPep8Naming
def get_dataframes():
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    return X, y
