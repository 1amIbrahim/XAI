# Owner: Salman Ali Khan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .base import BasePreprocessor
from config import ADULT_PATH, ADULT_COLUMNS, ADULT_TARGET

# Occupations with income>50K rate >= 40%
_HIGH_INCOME_OCC = {"Exec-managerial", "Prof-specialty", "Protective-serv", "Tech-support"}
_MARRIED_STATUS  = {"Married-civ-spouse", "Married-AF-spouse"}


class AdultPreprocessor(BasePreprocessor):
    """Preprocessor for the UCI Adult (Census Income) dataset.

    Target : income — binary (0 = <=50K, 1 = >50K)
    Fairness: sex, race, age_group

    Feature audit (23 → 16):
    DROPPED — education       : exact 1-to-1 duplicate of education-num
    DROPPED — native-country  : 41 cats, only 0.061 US/non-US spread; replaced by is_us_native
    DROPPED — capital-gain    : raw; replaced by capital_gain_log (91.6 % zeros → extreme skew)
    DROPPED — capital-loss    : raw; replaced by capital_loss_log
    DROPPED — capital_net     : gain_log + loss_log already encode this separately
    DROPPED — has_capital     : log1p(0)=0 already signals no-capital rows
    DROPPED — hours_bin       : redundant with continuous hours-per-week
    """

    def __init__(self):
        super().__init__(ADULT_PATH)
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # 6 categoricals  (education + native-country removed)
        self.categorical_cols = [
            "workclass", "marital-status", "occupation",
            "relationship", "race", "sex",
        ]
        # 10 numericals  (raw capital cols replaced by log transforms)
        self.numerical_cols = [
            "age", "education-num", "hours-per-week",
            "capital_gain_log", "capital_loss_log",
            "age_group", "age_edu_interact",
            "is_married", "is_us_native", "is_high_occ",
        ]

    # ── Data loading ──────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(
            self.data_path,
            header=None,
            names=ADULT_COLUMNS,
            sep=",",
            skipinitialspace=True,
        )
        self.df.drop(columns=["fnlwgt"], inplace=True)
        print(f"Loaded {self.data_path} — shape: {self.df.shape}")
        return self.df

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def clean(self) -> pd.DataFrame:
        self.df.replace("?", pd.NA, inplace=True)
        self.df.dropna(inplace=True)
        self.df[ADULT_TARGET] = self.df[ADULT_TARGET].str.strip()
        return self.df

    def feature_engineer(self) -> pd.DataFrame:
        """
        Add 7 engineered features then drop 7 noisy/redundant originals.
        Must run BEFORE encode() so raw string values are still available.
        """
        # Log-transform capital (handles 91.6 % zeros and right-skew outliers)
        self.df["capital_gain_log"] = np.log1p(self.df["capital-gain"])
        self.df["capital_loss_log"] = np.log1p(self.df["capital-loss"])

        # Ordinal age group — Young <40 / Middle 40-60 / Senior >60
        self.df["age_group"] = pd.cut(
            self.df["age"], bins=[-1, 39, 60, 100], labels=[0, 1, 2]
        ).astype(int)

        # Interaction: experience × education level
        self.df["age_edu_interact"] = (self.df["age"] * self.df["education-num"]) / 100.0

        # Binary flags from raw strings
        self.df["is_married"]   = self.df["marital-status"].isin(_MARRIED_STATUS).astype(int)
        self.df["is_us_native"] = (self.df["native-country"] == "United-States").astype(int)
        self.df["is_high_occ"]  = self.df["occupation"].isin(_HIGH_INCOME_OCC).astype(int)

        # Drop redundant/noisy columns
        self.df.drop(
            columns=["education", "native-country", "capital-gain", "capital-loss"],
            inplace=True,
        )
        return self.df

    def encode(self) -> pd.DataFrame:
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        self.df[ADULT_TARGET] = (self.df[ADULT_TARGET] == ">50K").astype(int)
        return self.df

    def scale(self) -> pd.DataFrame:
        self.df[self.numerical_cols] = self.scaler.fit_transform(
            self.df[self.numerical_cols]
        )
        return self.df

    def run(self, target_col: str):
        """Full pipeline: load → clean → feature_engineer → encode → scale → split."""
        self.load()
        self.clean()
        self.feature_engineer()
        self.encode()
        self.scale()
        return self.split(target_col)

    # ── Inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _age_group(age: float) -> int:
        return 0 if age < 40 else (1 if age <= 60 else 2)

    def transform_input(self, features_dict: dict) -> pd.DataFrame:
        """
        Apply all fitted transformers to a single raw feature dict for inference.
        Accepts the full original field set; dropped fields are used only for
        computing the engineered replacements.
        """
        # Columns that survive into the model (originals not dropped)
        kept_base = [
            "age", "workclass", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "hours-per-week",
        ]
        row = {col: features_dict.get(col, 0) for col in kept_base}
        df  = pd.DataFrame([row])

        # Raw values needed only for engineering (dropped from final feature set)
        cg      = float(features_dict.get("capital-gain", 0))
        cl      = float(features_dict.get("capital-loss", 0))
        age_val = float(features_dict.get("age", 30))
        edu_num = float(features_dict.get("education-num", 9))
        marital = str(features_dict.get("marital-status", "")).strip()
        country = str(features_dict.get("native-country", "")).strip()
        occ     = str(features_dict.get("occupation", "")).strip()

        # Engineered features
        df["capital_gain_log"] = np.log1p(cg)
        df["capital_loss_log"] = np.log1p(cl)
        df["age_group"]        = self._age_group(age_val)
        df["age_edu_interact"] = (age_val * edu_num) / 100.0
        df["is_married"]       = int(marital in _MARRIED_STATUS)
        df["is_us_native"]     = int(country == "United-States")
        df["is_high_occ"]      = int(occ in _HIGH_INCOME_OCC)

        # Encode categorical columns using fitted encoders
        for col in self.categorical_cols:
            if col in self.label_encoders:
                le  = self.label_encoders[col]
                val = str(df[col].iloc[0]).strip()
                df[col] = le.transform([val])[0] if val in le.classes_ else 0

        # Ensure numeric dtype for all numerical columns
        for col in self.numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Scale then return columns in the exact training order
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        return df[list(self.X_train.columns)]
