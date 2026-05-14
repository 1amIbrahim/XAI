import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
)


MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "neural_network": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}


class ModelTrainer:
    def __init__(self, dataset_name: str, save_dir: str = "saved_models"):
        self.dataset_name = dataset_name
        self.save_dir = save_dir
        self.trained = {}
        os.makedirs(save_dir, exist_ok=True)

    def train_all(self, X_train, y_train) -> dict:
        for name, factory in MODELS.items():
            model = factory()
            model.fit(X_train, y_train)
            self.trained[name] = model
            print(f"  Trained: {name}")
        return self.trained

    def evaluate_all(self, X_test, y_test) -> dict:
        results = {}
        for name, model in self.trained.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                "accuracy": acc,
                "report": classification_report(y_test, y_pred),
            }
            print(f"  {name}: accuracy={acc:.4f}")
        return results

    def save_all(self):
        """Save all trained models as .pkl files.
        Naming convention: saved_models/<dataset_name>_<model_name>.pkl
        """
        for name, model in self.trained.items():
            path = os.path.join(self.save_dir, f"{self.dataset_name}_{name}.pkl")
            joblib.dump(model, path)
            print(f"  Saved: {path}")

    def plot_results(self, X_test, y_test, vis_dir: str):
        """Generate and save training visualizations to vis_dir."""
        os.makedirs(vis_dir, exist_ok=True)
        _COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
        label = self.dataset_name.replace("_", " ").title()

        # 1. Accuracy comparison bar chart
        names = list(self.trained.keys())
        accs = [accuracy_score(y_test, m.predict(X_test)) for m in self.trained.values()]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(names, accs, color=_COLORS)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{label} — Model Accuracy Comparison")
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=11)
        ax.set_xticklabels([n.replace("_", "\n") for n in names])
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "accuracy_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Confusion matrices (2×2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, (name, model) in enumerate(self.trained.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(cm)
            disp.plot(ax=axes[i], colorbar=False)
            axes[i].set_title(name.replace("_", " ").title())
        plt.suptitle(f"{label} — Confusion Matrices", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 3. ROC curves (all models on one plot)
        fig, ax = plt.subplots(figsize=(8, 6))
        for (name, model), color in zip(self.trained.items(), _COLORS):
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, proba)
                area = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={area:.3f})", color=color)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{label} — ROC Curves")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "roc_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 4. Feature importances for tree-based models (top 15 features)
        feature_names = (
            list(X_test.columns) if hasattr(X_test, "columns")
            else [f"f{i}" for i in range(X_test.shape[1])]
        )
        for name in ("decision_tree", "random_forest"):
            if name not in self.trained:
                continue
            importances = self.trained[name].feature_importances_
            idx = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(idx)), importances[idx], color="steelblue")
            ax.set_xticks(range(len(idx)))
            ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
            ax.set_ylabel("Importance")
            ax.set_title(f"{label} — {name.replace('_', ' ').title()} Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{name}_feature_importance.png"), dpi=150, bbox_inches="tight")
            plt.close()

        # 5. Neural network training loss curve
        nn = self.trained.get("neural_network")
        if nn is not None and hasattr(nn, "loss_curve_"):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(nn.loss_curve_, color="steelblue")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title(f"{label} — Neural Network Training Loss")
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "neural_network_loss_curve.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"  Visualizations saved to {vis_dir}/")

    @staticmethod
    def load(dataset_name: str, model_name: str, save_dir: str = "saved_models"):
        """Load a saved model by dataset and model name."""
        path = os.path.join(save_dir, f"{dataset_name}_{model_name}.pkl")
        return joblib.load(path)
