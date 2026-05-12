import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

HEART_DISEASE_PATH = os.path.join(DATA_DIR, "heart_disease_data.csv")
ADULT_PATH = os.path.join(DATA_DIR, "adult", "adult.data")

HEART_DISEASE_TARGET = "target"
ADULT_TARGET = "income"

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

# Columns used for fairness analysis
ADULT_SENSITIVE_COLS = ["sex", "race", "age_group"]

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "neural_network"]
