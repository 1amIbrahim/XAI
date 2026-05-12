import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


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

    @staticmethod
    def load(dataset_name: str, model_name: str, save_dir: str = "saved_models"):
        """Load a saved model by dataset and model name."""
        path = os.path.join(save_dir, f"{dataset_name}_{model_name}.pkl")
        return joblib.load(path)
