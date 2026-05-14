# Owner: Rabiya Tahir
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.preprocessing.heart_disease import HeartDiseasePreprocessor
from src.models.base import ModelTrainer
from config import HEART_DISEASE_TARGET, VISUALIZATIONS_DIR


def main():
    print("=== Heart Disease: Preprocessing ===")
    preprocessor = HeartDiseasePreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run(target_col=HEART_DISEASE_TARGET)

    print("\n=== Heart Disease: Training Models ===")
    trainer = ModelTrainer(dataset_name="heart_disease")
    trainer.train_all(X_train, y_train)

    print("\n=== Heart Disease: Evaluation ===")
    trainer.evaluate_all(X_test, y_test)

    print("\n=== Heart Disease: Saving Models ===")
    trainer.save_all()

    print("\n=== Heart Disease: Saving Visualizations ===")
    vis_dir = os.path.join(VISUALIZATIONS_DIR, "training", "heart_disease")
    trainer.plot_results(X_test, y_test, vis_dir)

    print("\nDone. Models saved to saved_models/")


if __name__ == "__main__":
    main()
