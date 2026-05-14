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
