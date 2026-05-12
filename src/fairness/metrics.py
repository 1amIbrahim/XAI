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
        # Use only the features the model was trained on (ignores added sensitive columns)
        if hasattr(model, "feature_names_in_"):
            predict_cols = list(model.feature_names_in_)
            self.y_pred = model.predict(X_test[predict_cols])
        else:
            self.y_pred = model.predict(X_test)

    def demographic_parity(self, sensitive_col: str) -> dict:
        """Positive prediction rate per group — equal rates = fair."""
        result = {}
        for group in self.X_test[sensitive_col].unique():
            mask = self.X_test[sensitive_col] == group
            result[group] = self.y_pred[mask].mean()
        return result

    def equal_opportunity(self, sensitive_col: str) -> dict:
        """True positive rate (recall) per group — equal TPR = fair."""
        result = {}
        y_test_arr = np.array(self.y_test)
        y_pred_arr = np.array(self.y_pred)
        for group in self.X_test[sensitive_col].unique():
            mask = (self.X_test[sensitive_col] == group).values
            positives = y_test_arr[mask] == 1
            if positives.sum() == 0:
                result[group] = float("nan")
            else:
                result[group] = y_pred_arr[mask][positives].mean()
        return result

    def disparate_impact(self, sensitive_col: str) -> float:
        """Ratio of lowest to highest positive prediction rate (80% rule: <0.8 = bias)."""
        rates = self.demographic_parity(sensitive_col)
        values = [v for v in rates.values() if not np.isnan(v) and v > 0]
        if len(values) < 2:
            return float("nan")
        return min(values) / max(values)

    def full_report(self, sensitive_cols: list) -> pd.DataFrame:
        """Compile all three metrics for every group across all sensitive columns."""
        rows = []
        for col in sensitive_cols:
            dp = self.demographic_parity(col)
            eo = self.equal_opportunity(col)
            di = self.disparate_impact(col)
            for group in dp:
                rows.append({
                    "sensitive_col": col,
                    "group": group,
                    "demographic_parity": round(dp[group], 4),
                    "equal_opportunity": round(eo.get(group, float("nan")), 4),
                    "disparate_impact": round(di, 4),
                })
        return pd.DataFrame(rows)
