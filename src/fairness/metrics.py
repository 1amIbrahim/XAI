# Owner: Rabiya Tahir
import numpy as np
import pandas as pd


class FairnessAnalyzer:
    """Evaluate prediction fairness across demographic groups.

    Designed for use with both datasets:
      - Heart Disease: sensitive cols = ['sex', 'age_group']
      - Adult Income:  sensitive cols = ['sex', 'race', 'age_group']
    """

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        self.model = model
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.y_pred = model.predict(X_test)

    def demographic_parity(self, sensitive_col: str) -> dict:
        """Positive prediction rate per group — equal rates = fair.

        TODO (Rabiya): for each unique value in sensitive_col, compute
        fraction of positive predictions and return as dict.
        Example output: {"Male": 0.31, "Female": 0.11}
        Interpretation: large gap = model predicts positively more for one group.
        """
        raise NotImplementedError("Rabiya: implement demographic_parity()")

    def equal_opportunity(self, sensitive_col: str) -> dict:
        """True positive rate (recall) per group — equal TPR = fair.

        TODO (Rabiya): for each group, compute TPR = TP / (TP + FN).
        Example output: {"Male": 0.79, "Female": 0.53}
        """
        raise NotImplementedError("Rabiya: implement equal_opportunity()")

    def disparate_impact(self, sensitive_col: str) -> float:
        """Ratio of lowest to highest positive prediction rate.

        TODO (Rabiya): use demographic_parity() result.
        ratio = min_rate / max_rate
        Value < 0.8 is considered discriminatory (80% rule).
        """
        raise NotImplementedError("Rabiya: implement disparate_impact()")

    def full_report(self, sensitive_cols: list) -> pd.DataFrame:
        """Compile all metrics for all sensitive columns into one DataFrame.

        TODO (Rabiya): iterate sensitive_cols, call all three methods,
        return a DataFrame with columns:
          [sensitive_col, group, demographic_parity, equal_opportunity, disparate_impact]
        """
        raise NotImplementedError("Rabiya: implement full_report()")
