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
        self.X_test = X_test.copy().reset_index(drop=True)
        self.y_test = np.array(y_test).flatten()
        self.y_pred = model.predict(X_test)

    def demographic_parity(self, sensitive_col: str) -> dict:
        """Positive prediction rate per group — equal rates = fair."""
        result = {}
        for group in sorted(self.X_test[sensitive_col].unique()):
            mask = (self.X_test[sensitive_col] == group).values
            result[int(group)] = float(self.y_pred[mask].mean())
        return result

    def equal_opportunity(self, sensitive_col: str) -> dict:
        """True positive rate (recall) per group — equal TPR = fair."""
        result = {}
        for group in sorted(self.X_test[sensitive_col].unique()):
            mask = (self.X_test[sensitive_col] == group).values
            y_true_g = self.y_test[mask]
            y_pred_g = self.y_pred[mask]
            tp = int(((y_pred_g == 1) & (y_true_g == 1)).sum())
            fn = int(((y_pred_g == 0) & (y_true_g == 1)).sum())
            result[int(group)] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return result

    def disparate_impact(self, sensitive_col: str) -> float:
        """min_rate / max_rate — below 0.8 indicates bias (80% rule)."""
        rates = list(self.demographic_parity(sensitive_col).values())
        if not rates or max(rates) == 0:
            return 0.0
        return float(min(rates) / max(rates))

    def full_report(self, sensitive_cols: list) -> pd.DataFrame:
        """Summary DataFrame for all metrics x all sensitive columns."""
        rows = []
        for col in sensitive_cols:
            dp = self.demographic_parity(col)
            eop = self.equal_opportunity(col)
            di = self.disparate_impact(col)
            for group in dp:
                rows.append({
                    "sensitive_col": col,
                    "group": group,
                    "demographic_parity": round(dp[group], 4),
                    "equal_opportunity": round(eop.get(group, 0.0), 4),
                    "disparate_impact": round(di, 4),
                })
        return pd.DataFrame(rows)
