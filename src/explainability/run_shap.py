# Owner: Muhammad Ibrahim
"""
Run SHAP explainability on all saved models for both datasets.

Usage (from project root):
    python src/explainability/run_shap.py

Prerequisites:
    - Phase 1 training complete: saved_models/ must contain all 8 .pkl files
    - pip install -r requirements.txt

Output:
    reports/shap/heart_disease/<model>_summary.png
    reports/shap/heart_disease/<model>_bar.png
    reports/shap/heart_disease/<model>_waterfall.png
    reports/shap/heart_disease/<model>_dependence_<feature>.png
    reports/shap/adult/  (same structure)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.preprocessing.heart_disease import HeartDiseasePreprocessor
from src.preprocessing.adult import AdultPreprocessor
from src.explainability.shap_explainer import SHAPExplainer
from config import HEART_DISEASE_TARGET, ADULT_TARGET, MODEL_NAMES


DATASETS = {
    "heart_disease": {
        "preprocessor_cls": HeartDiseasePreprocessor,
        "target": HEART_DISEASE_TARGET,
    },
    "adult": {
        "preprocessor_cls": AdultPreprocessor,
        "target": ADULT_TARGET,
    },
}


def run_dataset(dataset_name: str, cfg: dict):
    print(f"\n{'='*55}")
    print(f"  Dataset: {dataset_name}")
    print(f"{'='*55}")

    preprocessor = cfg["preprocessor_cls"]()
    X_train, X_test, y_train, y_test = preprocessor.run(target_col=cfg["target"])

    errors = []
    for model_name in MODEL_NAMES:
        try:
            explainer = SHAPExplainer(dataset_name, model_name, X_train)
            explainer.run_all(X_test)
        except FileNotFoundError as e:
            print(f"  SKIP  [{model_name}]: {e}")
            errors.append(model_name)
        except Exception as e:
            print(f"  ERROR [{model_name}]: {type(e).__name__}: {e}")
            errors.append(model_name)

    return errors


def main():
    print("SHAP Explainability Runner")
    print("Ibrahim — Phase 2\n")

    all_errors = {}
    for dataset_name, cfg in DATASETS.items():
        errs = run_dataset(dataset_name, cfg)
        if errs:
            all_errors[dataset_name] = errs

    print("\n" + "="*55)
    if all_errors:
        print("Completed with errors:")
        for ds, models in all_errors.items():
            print(f"  {ds}: {models}")
    else:
        print("All SHAP plots generated successfully.")
    print("Output: reports/shap/")


if __name__ == "__main__":
    main()
