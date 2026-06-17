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
        # Fill missing values with median (only ca and thal have 6 missing rows total)
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return self.df

    def encode(self) -> pd.DataFrame:
        # All columns in the UCI Heart Disease dataset are numeric — nothing to encode
        return self.df

    def scale(self) -> pd.DataFrame:
        self.feature_cols = [c for c in self.df.columns if c != "target"]
        self.df[self.feature_cols] = self.scaler.fit_transform(self.df[self.feature_cols])
        return self.df

    def transform_input(self, features_dict: dict) -> pd.DataFrame:
        """Apply the fitted scaler to one raw patient feature dictionary."""
        if self.feature_cols is None:
            self.feature_cols = [c for c in self.X_train.columns]

        row = {
            col: pd.to_numeric(features_dict.get(col, 0), errors="coerce")
            for col in self.feature_cols
        }
        df = pd.DataFrame([row]).fillna(0.0)
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        return df[list(self.X_train.columns)]
