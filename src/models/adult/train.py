# Owner: Salman Ali Khan
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from src.preprocessing.adult import AdultPreprocessor
from src.models.base import ModelTrainer
from config import ADULT_TARGET, VISUALIZATIONS_DIR

# Tuned models for the Adult Income dataset
ADULT_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
    "neural_network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}


def main():
    print("=== Adult Income: Preprocessing ===")
    preprocessor = AdultPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run(target_col=ADULT_TARGET)

    print("\n=== Adult Income: Training Models ===")
    trainer = ModelTrainer(dataset_name="adult")
    for name, model in ADULT_MODELS.items():
        model.fit(X_train, y_train)
        trainer.trained[name] = model
        print(f"  Trained: {name}")

    print("\n=== Adult Income: Evaluation ===")
    trainer.evaluate_all(X_test, y_test)

    print("\n=== Adult Income: Saving Models ===")
    trainer.save_all()

    print("\n=== Adult Income: Saving Visualizations ===")
    vis_dir = os.path.join(VISUALIZATIONS_DIR, "training", "adult")
    trainer.plot_results(X_test, y_test, vis_dir)

    print("\nDone. Models saved to saved_models/")


if __name__ == "__main__":
    main()
