from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split


class BasePreprocessor(ABC):
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {self.data_path} — shape: {self.df.shape}")
        return self.df

    @abstractmethod
    def clean(self) -> pd.DataFrame:
        """Handle missing values, drop duplicates, fix data types."""

    @abstractmethod
    def encode(self) -> pd.DataFrame:
        """Encode categorical variables."""

    @abstractmethod
    def scale(self) -> pd.DataFrame:
        """Scale / normalise numerical features."""

    def split(self, target_col: str, test_size: float = 0.2, random_state: int = 42):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train: {self.X_train.shape}  Test: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def run(self, target_col: str):
        """Full pipeline: load → clean → encode → scale → split."""
        self.load()
        self.clean()
        self.encode()
        self.scale()
        return self.split(target_col)
