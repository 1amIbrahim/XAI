# Owner: Rabiya Tahir
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .base import BasePreprocessor
from config import HEART_DISEASE_PATH


class HeartDiseasePreprocessor(BasePreprocessor):
    def __init__(self):
        super().__init__(HEART_DISEASE_PATH)
        self.scaler = StandardScaler()
        self.feature_cols = None

    def clean(self) -> pd.DataFrame:
        # TODO (Rabiya): check for missing values and handle them
        # Hint: self.df.isnull().sum() to inspect, then drop or fill
        # self.df = self.df.dropna()
        raise NotImplementedError("Rabiya: implement clean()")

    def encode(self) -> pd.DataFrame:
        # TODO (Rabiya): encode any categorical columns if needed
        # The Heart Disease dataset from UCI is mostly numeric already.
        # Check df.dtypes and apply pd.get_dummies or LabelEncoder if required.
        raise NotImplementedError("Rabiya: implement encode()")

    def scale(self) -> pd.DataFrame:
        # TODO (Rabiya): scale all numeric features except the target column
        # Example:
        #   self.feature_cols = [c for c in self.df.columns if c != "target"]
        #   self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        raise NotImplementedError("Rabiya: implement scale()")
