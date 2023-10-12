import pandas.core.frame


def preprocess_heart_diseases(df: pandas.core.frame.DataFrame):
    return df.drop(['ca', 'thal'], axis=1)
