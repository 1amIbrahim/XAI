# Owner: Salman Ali Khan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .base import BasePreprocessor
from config import ADULT_PATH, ADULT_COLUMNS, ADULT_TARGET


class AdultPreprocessor(BasePreprocessor):
    """Preprocessor for the UCI Adult (Census Income) dataset.

    Target: income — binary (0 = <=50K, 1 = >50K)
    Sensitive columns for fairness: sex, race, age_group
    """

    def __init__(self):
        super().__init__(ADULT_PATH)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_cols = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ]
        self.numerical_cols = [
            "age", "education-num", "capital-gain", "capital-loss", "hours-per-week",
        ]

    def load(self) -> pd.DataFrame:
        # adult.data has no header and uses ", " as separator
        self.df = pd.read_csv(
            self.data_path,
            header=None,
            names=ADULT_COLUMNS,
            sep=",",
            skipinitialspace=True,
        )
        # Drop fnlwgt — census sampling weight, not a predictive feature
        self.df.drop(columns=["fnlwgt"], inplace=True)
        print(f"Loaded {self.data_path} — shape: {self.df.shape}")
        return self.df

    def clean(self) -> pd.DataFrame:
        # TODO (Salman): handle missing values encoded as "?"
        # Step 1: replace "?" with NaN
        #   self.df.replace("?", pd.NA, inplace=True)
        # Step 2: drop rows with NaN (or fill with mode)
        #   self.df.dropna(inplace=True)
        # Step 3: strip whitespace from target column
        #   self.df[ADULT_TARGET] = self.df[ADULT_TARGET].str.strip()
        raise NotImplementedError("Salman: implement clean()")

    def encode(self) -> pd.DataFrame:
        # TODO (Salman): encode categorical columns with LabelEncoder
        # Also binarise the target: <=50K → 0, >50K → 1
        # Example:
        #   for col in self.categorical_cols:
        #       le = LabelEncoder()
        #       self.df[col] = le.fit_transform(self.df[col].astype(str))
        #       self.label_encoders[col] = le
        #   self.df[ADULT_TARGET] = (self.df[ADULT_TARGET] == ">50K").astype(int)
        raise NotImplementedError("Salman: implement encode()")

    def scale(self) -> pd.DataFrame:
        # TODO (Salman): scale numerical columns with StandardScaler
        # Example:
        #   self.df[self.numerical_cols] = self.scaler.fit_transform(
        #       self.df[self.numerical_cols]
        #   )
        raise NotImplementedError("Salman: implement scale()")
