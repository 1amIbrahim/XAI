# Owner: Salman Ali Khan
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.preprocessing.adult import AdultPreprocessor
from src.models.base import ModelTrainer
from config import ADULT_TARGET


def main():
    print("=== Adult Income: Preprocessing ===")
    preprocessor = AdultPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run(target_col=ADULT_TARGET)

    print("\n=== Adult Income: Training Models ===")
    trainer = ModelTrainer(dataset_name="adult")
    trainer.train_all(X_train, y_train)

    print("\n=== Adult Income: Evaluation ===")
    trainer.evaluate_all(X_test, y_test)

    print("\n=== Adult Income: Saving Models ===")
    trainer.save_all()
    print("\nDone. Models saved to saved_models/")


if __name__ == "__main__":
    main()
