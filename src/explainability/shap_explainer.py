# Owner: Muhammad Ibrahim
import os
import shap
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SAVED_MODELS_DIR, REPORTS_DIR, MODEL_NAMES


class SHAPExplainer:
    def __init__(self, dataset_name: str, model_name: str, X_train: pd.DataFrame):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.X_train = X_train
        self.model = self._load_model()
        self.explainer = None
        self.shap_values = None
        self.output_dir = os.path.join(REPORTS_DIR, "shap", dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self):
        path = os.path.join(SAVED_MODELS_DIR, f"{self.dataset_name}_{self.model_name}.pkl")
        return joblib.load(path)

    def fit(self):
        if self.model_name in ("random_forest", "decision_tree"):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, shap.sample(self.X_train, 100)
            )
        return self

    def compute(self, X_test: pd.DataFrame):
        self.shap_values = self.explainer(X_test)
        return self.shap_values

    def summary_plot(self, X_test: pd.DataFrame):
        shap.summary_plot(self.shap_values, X_test, show=False)
        path = os.path.join(self.output_dir, f"{self.model_name}_summary.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {path}")

    def waterfall_plot(self, index: int = 0):
        shap.plots.waterfall(self.shap_values[index], show=False)
        path = os.path.join(self.output_dir, f"{self.model_name}_waterfall_{index}.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {path}")

    def dependence_plot(self, feature: str, X_test: pd.DataFrame):
        shap.dependence_plot(feature, self.shap_values.values, X_test, show=False)
        path = os.path.join(self.output_dir, f"{self.model_name}_dependence_{feature}.png")
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved: {path}")

    def run_all(self, X_test: pd.DataFrame, top_feature: str = None):
        """Generate all three plot types for this model."""
        self.fit()
        self.compute(X_test)
        self.summary_plot(X_test)
        self.waterfall_plot(index=0)
        if top_feature:
            self.dependence_plot(top_feature, X_test)
