# Dataset implementation
from preprocess import preprocess_heart_diseases
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = preprocess_heart_diseases(heart_disease.data.features)
y = heart_disease.data.targets
