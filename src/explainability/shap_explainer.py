# Owner: Muhammad Ibrahim
import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SAVED_MODELS_DIR, REPORTS_DIR


class SHAPExplainer:
    """SHAP explainability for all 4 model types on both datasets.

    Handles three explainer backends automatically:
      - TreeExplainer  → Random Forest, Decision Tree  (fast, exact)
      - LinearExplainer → Logistic Regression          (fast, exact)
      - KernelExplainer → Neural Network (MLP)          (slow, model-agnostic)

    All plot methods operate on the positive class (class 1) SHAP values so
    that summary/waterfall/dependence plots have a consistent interpretation
    across both tree and linear models.
    """

    def __init__(self, dataset_name: str, model_name: str, X_train: pd.DataFrame):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.X_train = X_train.reset_index(drop=True)
        self.feature_names = list(X_train.columns)
        self.model = self._load_model()
        self.explainer = None
        self.shap_values = None   # 2-D array (n_samples, n_features), positive class
        self.base_value = None    # scalar expected value for positive class
        self.output_dir = os.path.join(REPORTS_DIR, "shap", dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model(self):
        path = os.path.join(SAVED_MODELS_DIR, f"{self.dataset_name}_{self.model_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                "Run the Phase 1 training script first."
            )
        return joblib.load(path)

    def fit(self):
        """Build the appropriate SHAP explainer for this model type."""
        if self.model_name in ("random_forest", "decision_tree"):
            self.explainer = shap.TreeExplainer(self.model)

        elif self.model_name == "logistic_regression":
            self.explainer = shap.LinearExplainer(self.model, self.X_train)

        else:  # neural_network — KernelExplainer, use a small background sample
            background = self.X_train.sample(100, random_state=42)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
        return self

    # ------------------------------------------------------------------
    # Compute SHAP values
    # ------------------------------------------------------------------

    # Maximum test rows for each explainer type — keeps runtime reasonable
    _MAX_SAMPLES = {
        "logistic_regression": 2000,
        "decision_tree":       500,
        "random_forest":       500,
        "neural_network":      150,
    }

    def compute(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values and return the (possibly sampled) test slice used.

        Caps test-set size per model type to keep runtime reasonable:
          - Random Forest / Decision Tree : 500 rows  (TreeExplainer is O(n*trees))
          - Logistic Regression           : 2000 rows (LinearExplainer is fast)
          - Neural Network                : 150 rows  (KernelExplainer is very slow)

        Normalises output to a 2-D array (n_samples, n_features) for class 1.
        """
        X_test = X_test.reset_index(drop=True)
        max_n = self._MAX_SAMPLES[self.model_name]
        X_sample = (
            X_test.sample(max_n, random_state=42).reset_index(drop=True)
            if len(X_test) > max_n else X_test
        )

        if self.model_name == "neural_network":
            raw = self.explainer.shap_values(X_sample, nsamples=100, silent=True)
        else:
            raw = self.explainer.shap_values(X_sample)

        n_samples, n_features = X_sample.shape[0], len(self.feature_names)
        raw_arr = np.array(raw)

        if raw_arr.ndim == 3:
            # Modern SHAP: (n_samples, n_features, n_classes)  — shape[-1] == 2
            if raw_arr.shape == (n_samples, n_features, 2):
                self.shap_values = raw_arr[:, :, 1]
            # Legacy SHAP: (n_classes, n_samples, n_features) — shape[0] == 2
            elif raw_arr.shape == (2, n_samples, n_features):
                self.shap_values = raw_arr[1]
            else:
                raise ValueError(f"Unexpected SHAP values shape: {raw_arr.shape}")
            ev = self.explainer.expected_value
            self.base_value = float(ev[1]) if hasattr(ev, "__len__") else float(ev)

        else:
            # 2-D: LinearExplainer for binary → (n_samples, n_features)
            self.shap_values = raw_arr
            ev = self.explainer.expected_value
            self.base_value = float(ev) if np.isscalar(ev) else float(ev[0])

        return X_sample

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _top_feature_names(self, n: int = 2) -> list:
        """Return names of top-n features ranked by mean |SHAP value|."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:n]
        return [self.feature_names[i] for i in idx]

    def _save(self, filename: str):
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"    Saved: {path}")

    # ------------------------------------------------------------------
    # Individual plots
    # ------------------------------------------------------------------

    def summary_plot(self, X_sample: pd.DataFrame):
        """Beeswarm plot — global importance with feature value colouring."""
        shap.summary_plot(
            self.shap_values, X_sample,
            feature_names=self.feature_names,
            show=False,
        )
        self._save(f"{self.model_name}_summary.png")

    def bar_plot(self, X_sample: pd.DataFrame):
        """Bar chart of mean |SHAP| — clean global importance ranking."""
        shap.summary_plot(
            self.shap_values, X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )
        self._save(f"{self.model_name}_bar.png")

    def waterfall_plot(self, X_sample: pd.DataFrame, index: int = 0):
        """Waterfall plot for one instance — local explanation."""
        exp = shap.Explanation(
            values=self.shap_values[index],
            base_values=self.base_value,
            data=X_sample.iloc[index].values,
            feature_names=self.feature_names,
        )
        shap.plots.waterfall(exp, show=False)
        self._save(f"{self.model_name}_waterfall.png")

    def dependence_plot(self, feature: str, X_sample: pd.DataFrame):
        """Dependence plot — marginal effect of one feature (with auto interaction)."""
        if feature not in self.feature_names:
            print(f"    Skipped dependence: '{feature}' not found in features")
            return
        shap.dependence_plot(
            feature, self.shap_values, X_sample,
            feature_names=self.feature_names,
            show=False,
        )
        safe = feature.replace("-", "_")
        self._save(f"{self.model_name}_dependence_{safe}.png")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_all(self, X_test: pd.DataFrame):
        """Fit → compute → generate all 4 plot types for this model."""
        print(f"  [{self.dataset_name}] {self.model_name}")
        self.fit()
        X_sample = self.compute(X_test)
        self.summary_plot(X_sample)
        self.bar_plot(X_sample)
        self.waterfall_plot(X_sample, index=0)
        for feat in self._top_feature_names(n=2):
            self.dependence_plot(feat, X_sample)
