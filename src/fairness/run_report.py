# Owner: Rabiya Tahir
"""
Generate fairness analysis reports and charts for heart disease dataset.
Run from project root:  python src/fairness/run_report.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.preprocessing.heart_disease import HeartDiseasePreprocessor
from src.models.base import ModelTrainer
from src.fairness.metrics import FairnessAnalyzer
from config import HEART_DISEASE_PATH, HEART_DISEASE_TARGET, REPORTS_DIR


def build_sensitive_cols(X_test: pd.DataFrame) -> pd.DataFrame:
    """Re-attach readable sensitive columns (sex label + age_group) to scaled X_test."""
    df_orig = pd.read_csv(HEART_DISEASE_PATH)
    df_orig = df_orig.fillna(df_orig.median(numeric_only=True))
    df_orig = df_orig.drop_duplicates().reset_index(drop=True)

    # Bin age into three groups
    df_orig["age_group"] = pd.cut(
        df_orig["age"],
        bins=[0, 39, 60, 200],
        labels=["Young (<40)", "Middle (40-60)", "Senior (>60)"],
    )
    df_orig["sex_label"] = df_orig["sex"].map({0: "Female", 1: "Male"})

    sensitive = df_orig.loc[X_test.index, ["sex_label", "age_group"]].copy()

    X_fair = X_test.copy()
    X_fair["sex"] = sensitive["sex_label"].values
    X_fair["age_group"] = sensitive["age_group"].values
    return X_fair


def plot_metric(report: pd.DataFrame, metric: str, dataset: str, save_dir: str):
    """Bar chart: metric value per group, grouped by sensitive column."""
    fig, axes = plt.subplots(
        1, len(report["sensitive_col"].unique()),
        figsize=(5 * len(report["sensitive_col"].unique()), 5),
        squeeze=False,
    )
    for idx, col in enumerate(report["sensitive_col"].unique()):
        sub = report[report["sensitive_col"] == col]
        ax = axes[0][idx]
        bars = ax.bar(sub["group"].astype(str), sub[metric], color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"])
        ax.set_title(f"{metric.replace('_', ' ').title()} — {col}", fontsize=11)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="0.8 threshold")
        ax.legend(fontsize=8)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{dataset.replace('_', ' ').title()} — {metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, f"{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def run_heart_disease():
    print("\n=== Fairness Analysis: Heart Disease ===")
    save_dir = os.path.join(REPORTS_DIR, "fairness", "heart_disease")
    os.makedirs(save_dir, exist_ok=True)

    prep = HeartDiseasePreprocessor()
    X_train, X_test, y_train, y_test = prep.run(target_col=HEART_DISEASE_TARGET)

    X_test_fair = build_sensitive_cols(X_test)

    sensitive_cols = ["sex", "age_group"]
    model_name = "random_forest"

    model = ModelTrainer.load("heart_disease", model_name)
    analyzer = FairnessAnalyzer(model, X_test_fair, y_test)

    report = analyzer.full_report(sensitive_cols)
    print(report.to_string(index=False))

    csv_path = os.path.join(save_dir, "fairness_report.csv")
    report.to_csv(csv_path, index=False)
    print(f"\nReport saved: {csv_path}")

    for metric in ["demographic_parity", "equal_opportunity"]:
        plot_metric(report, metric, "heart_disease", save_dir)

    # Disparate impact summary chart
    di_rows = report.drop_duplicates(subset=["sensitive_col"])[["sensitive_col", "disparate_impact"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(di_rows["sensitive_col"], di_rows["disparate_impact"], color=["#4e79a7", "#f28e2b"])
    ax.axhline(0.8, color="red", linestyle="--", linewidth=1.5, label="Fairness threshold (0.8)")
    ax.set_ylabel("Disparate Impact Ratio")
    ax.set_title("Disparate Impact — Heart Disease", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for i, row in di_rows.iterrows():
        ax.text(list(di_rows["sensitive_col"]).index(row["sensitive_col"]),
                row["disparate_impact"] + 0.03,
                f"{row['disparate_impact']:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "disparate_impact.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.join(save_dir, 'disparate_impact.png')}")


if __name__ == "__main__":
    run_heart_disease()
    print("\nFairness analysis complete.")
