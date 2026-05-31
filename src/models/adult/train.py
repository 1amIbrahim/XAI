# Owner: Salman Ali Khan
import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from src.preprocessing.adult import AdultPreprocessor
from src.models.base import ModelTrainer
from config import ADULT_TARGET, VISUALIZATIONS_DIR

# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

LR_PARAM_DIST = {
    "C":            [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0],
    "class_weight": [None, "balanced"],
    "penalty":      ["l1", "l2"],
}

DT_PARAM_DIST = {
    "max_depth":        [5, 7, 8, 10, 12, 15],
    "min_samples_split":[2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "criterion":        ["gini", "entropy"],
    "class_weight":     [None, "balanced"],
    "max_features":     [None, "sqrt"],
}

RF_PARAM_DIST = {
    "n_estimators":     [200, 300, 400, 500],
    "max_depth":        [10, 15, 20, None],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", 0.3],
    "class_weight":     [None, "balanced_subsample"],
}

NN_PARAM_DIST = {
    "hidden_layer_sizes": [
        (64, 32), (128, 64), (256, 128), (128, 64, 32), (256, 128, 64),
    ],
    "alpha":              [0.0001, 0.0005, 0.001, 0.01],
    "learning_rate_init": [0.0005, 0.001, 0.005],
    "activation":         ["relu", "tanh"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tune(estimator, param_dist, X_train, y_train, n_iter=30):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=n_iter, cv=cv, scoring="accuracy",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"    best params  : {search.best_params_}")
    print(f"    CV accuracy  : {search.best_score_:.4f}")
    return search.best_estimator_


def _acc(model, X_test, y_test):
    return accuracy_score(y_test, model.predict(X_test))


def _banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ── Step 1: Data & feature summary ───────────────────────────────────────
    _banner("Step 1 — Preprocessing & Feature Engineering")
    prep = AdultPreprocessor()
    X_train, X_test, y_train, y_test = prep.run(target_col=ADULT_TARGET)

    n_feat = len(X_train.columns)
    print(f"\n  Final feature count : {n_feat}")
    print(f"  Features            : {list(X_train.columns)}")
    print(f"  Train / Test        : {X_train.shape[0]} / {X_test.shape[0]}")
    pos = y_train.mean()
    print(f"  Class balance       : {pos:.1%} positive (>50K)")

    # ── Step 2: Baseline (no tuning) ─────────────────────────────────────────
    _banner("Step 2 — Baseline Accuracy (default hyper-parameters)")

    default_cfg = {
        "logistic_regression": LogisticRegression(
            max_iter=3000, solver="saga", random_state=42
        ),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "neural_network": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }
    baseline: dict[str, float] = {}
    for name, model in default_cfg.items():
        model.fit(X_train, y_train)
        acc = _acc(model, X_test, y_test)
        baseline[name] = acc
        print(f"  {name:<26}  {acc:.4f}")

    # ── Step 3: [1/4] Logistic Regression ────────────────────────────────────
    _banner("Step 3 — [1/4] Logistic Regression")
    print("  saga solver, tuning C + penalty + class_weight")
    best_lr = _tune(
        LogisticRegression(max_iter=3000, solver="saga", random_state=42),
        LR_PARAM_DIST, X_train, y_train, n_iter=30,
    )
    lr_acc = _acc(best_lr, X_test, y_test)
    delta  = lr_acc - baseline["logistic_regression"]
    print(f"    BEFORE: {baseline['logistic_regression']:.4f}  AFTER: {lr_acc:.4f}  ({delta:+.4f})")

    # ── Step 4: [2/4] Decision Tree ───────────────────────────────────────────
    _banner("Step 4 — [2/4] Decision Tree")
    print("  Tuning depth, leaf size, criterion, class_weight")
    best_dt = _tune(
        DecisionTreeClassifier(random_state=42),
        DT_PARAM_DIST, X_train, y_train, n_iter=40,
    )
    dt_acc = _acc(best_dt, X_test, y_test)
    delta  = dt_acc - baseline["decision_tree"]
    print(f"    BEFORE: {baseline['decision_tree']:.4f}  AFTER: {dt_acc:.4f}  ({delta:+.4f})")

    # ── Step 5: [3/4] Random Forest ───────────────────────────────────────────
    _banner("Step 5 — [3/4] Random Forest")
    print("  Tuning n_estimators, depth, features, class_weight")
    best_rf = _tune(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        RF_PARAM_DIST, X_train, y_train, n_iter=50,
    )
    rf_acc = _acc(best_rf, X_test, y_test)
    delta  = rf_acc - baseline["random_forest"]
    print(f"    BEFORE: {baseline['random_forest']:.4f}  AFTER: {rf_acc:.4f}  ({delta:+.4f})")

    # ── Step 6: [4/4] Neural Network ──────────────────────────────────────────
    _banner("Step 6 — [4/4] Neural Network")
    print("  Tuning architecture, activation, alpha, learning_rate_init")
    best_nn = _tune(
        MLPClassifier(max_iter=500, random_state=42),
        NN_PARAM_DIST, X_train, y_train, n_iter=25,
    )
    nn_acc = _acc(best_nn, X_test, y_test)
    delta  = nn_acc - baseline["neural_network"]
    print(f"    BEFORE: {baseline['neural_network']:.4f}  AFTER: {nn_acc:.4f}  ({delta:+.4f})")

    # ── Step 7: Final summary ─────────────────────────────────────────────────
    _banner("Step 7 — Final Accuracy Summary")
    final = {
        "logistic_regression": best_lr,
        "decision_tree":       best_dt,
        "random_forest":       best_rf,
        "neural_network":      best_nn,
    }
    print(f"\n  {'Model':<26}  {'Baseline':>9}  {'Tuned':>9}  {'Gain':>8}")
    print(f"  {'-'*56}")
    for name, model in final.items():
        after  = _acc(model, X_test, y_test)
        before = baseline[name]
        sign   = "+" if (after - before) >= 0 else ""
        print(f"  {name:<26}  {before:>9.4f}  {after:>9.4f}  {sign}{after - before:>7.4f}")

    print(f"\n  Best model: Random Forest  {rf_acc:.4f}")
    print("\n  Classification report (Random Forest):")
    print(classification_report(
        y_test, best_rf.predict(X_test),
        target_names=["<=50K", ">50K"], digits=4,
    ))

    # ── Feature importance (RF) ───────────────────────────────────────────────
    _banner("Step 8 — Feature Importance (Tuned Random Forest)")
    importances = best_rf.feature_importances_
    feat_imp = sorted(zip(X_train.columns, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 200)
        print(f"  {feat:<22}  {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _banner("Step 9 — Saving Models")
    trainer = ModelTrainer(dataset_name="adult")
    trainer.trained = final
    trainer.save_all()
    print("\n  Done. All 4 models saved to saved_models/")


if __name__ == "__main__":
    main()
