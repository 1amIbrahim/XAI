# Owner: Salman Ali Khan
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from config import ADULT_TARGET, ADULT_SENSITIVE_COLS, REPORTS_DIR

st.set_page_config(page_title="XAI Adult Income", layout="wide")

# ── Category constants ────────────────────────────────────────────────────────
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}
WORKCLASS = ["Federal-gov", "Local-gov", "Never-worked", "Private",
             "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"]
EDUCATION = list(EDUCATION_NUM.keys())
MARITAL = ["Divorced", "Married-AF-spouse", "Married-civ-spouse",
           "Married-spouse-absent", "Never-married", "Separated", "Widowed"]
OCCUPATION = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
              "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
              "Other-service", "Priv-house-serv", "Prof-specialty",
              "Protective-serv", "Sales", "Tech-support", "Transport-moving"]
RELATIONSHIP = ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
RACE = ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]
SEX = ["Female", "Male"]
COUNTRY = [
    "Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic",
    "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece",
    "Guatemala", "Haiti", "Honduras", "Hong", "Hungary", "India", "Iran",
    "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua",
    "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal",
    "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand",
    "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia",
]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "neural_network": "Neural Network",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading dataset and fitting preprocessor…")
def load_preprocessor():
    from src.preprocessing.adult import AdultPreprocessor
    prep = AdultPreprocessor()
    prep.run(ADULT_TARGET)
    return prep


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_name: str):
    from src.models.base import ModelTrainer
    return ModelTrainer.load("adult", model_name)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("XAI — Adult Income")
page = st.sidebar.selectbox("Page", ["Prediction", "Explanation", "Fairness"])

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Prediction":
    st.title("Income Prediction")
    st.markdown(
        "Enter an individual's details and choose a model to predict whether "
        "their income exceeds \\$50K/year."
    )

    model_label = st.selectbox("Model", list(MODEL_LABELS.values()))
    model_name = LABEL_TO_KEY[model_label]

    st.subheader("Individual Features")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35)
        workclass = st.selectbox("Workclass", WORKCLASS, index=WORKCLASS.index("Private"))
        education = st.selectbox("Education", EDUCATION, index=EDUCATION.index("Bachelors"))
        capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=99999, value=0)

    with c2:
        marital = st.selectbox("Marital Status", MARITAL,
                               index=MARITAL.index("Never-married"))
        occupation = st.selectbox("Occupation", OCCUPATION,
                                  index=OCCUPATION.index("Prof-specialty"))
        relationship = st.selectbox("Relationship", RELATIONSHIP,
                                    index=RELATIONSHIP.index("Not-in-family"))
        capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=99999, value=0)

    with c3:
        race = st.selectbox("Race", RACE, index=RACE.index("White"))
        sex = st.selectbox("Sex", SEX, index=SEX.index("Male"))
        hours = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
        country = st.selectbox("Native Country", COUNTRY,
                               index=COUNTRY.index("United-States"))

    if st.button("Predict", type="primary"):
        features = {
            "age": age,
            "workclass": workclass,
            "education": education,
            "education-num": EDUCATION_NUM[education],
            "marital-status": marital,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours,
            "native-country": country,
        }
        try:
            prep = load_preprocessor()
            model = load_model(model_name)
            X = prep.transform_input(features)
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            label = ">50K" if prediction == 1 else "<=50K"

            st.divider()
            col_r, col_p = st.columns([1, 2])
            with col_r:
                if prediction == 1:
                    st.success(f"Predicted Income: **{label}**")
                else:
                    st.warning(f"Predicted Income: **{label}**")
                st.metric("P(income > 50K)", f"{probability:.1%}")
            with col_p:
                st.markdown("**Confidence**")
                st.progress(probability)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explanation":
    st.title("SHAP Explanations")
    st.markdown(
        "Global summary plots show which features matter most overall. "
        "Waterfall plots show why a single prediction was made."
    )

    model_label = st.selectbox("Model", list(MODEL_LABELS.values()))
    model_name = LABEL_TO_KEY[model_label]

    shap_dir = Path(REPORTS_DIR) / "shap" / "adult"
    summary_path = shap_dir / f"{model_name}_summary.png"
    waterfall_path = shap_dir / f"{model_name}_waterfall_0.png"

    if not shap_dir.exists() or (not summary_path.exists() and not waterfall_path.exists()):
        st.info(
            "SHAP plots have not been generated yet for this model. "
            "Run `src/explainability/shap_explainer.py` (Ibrahim's module) first."
        )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Summary Plot — Global Feature Importance")
            if summary_path.exists():
                st.image(str(summary_path), use_container_width=True)
            else:
                st.warning("Summary plot not found.")
        with col2:
            st.subheader("Waterfall Plot — Local Explanation (sample 0)")
            if waterfall_path.exists():
                st.image(str(waterfall_path), use_container_width=True)
            else:
                st.warning("Waterfall plot not found.")

# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Fairness":
    st.title("Fairness Analysis")
    st.markdown(
        "Demographic parity = fraction predicted positive per group. "
        "Equal opportunity = true positive rate per group. "
        "Disparate impact < 0.8 signals potential bias (80% rule)."
    )

    model_label = st.selectbox("Model", list(MODEL_LABELS.values()))
    model_name = LABEL_TO_KEY[model_label]

    try:
        prep = load_preprocessor()
        model = load_model(model_name)
        from src.fairness.metrics import FairnessAnalyzer

        sensitive_cols = [c for c in ADULT_SENSITIVE_COLS if c in prep.X_test.columns]
        analyzer = FairnessAnalyzer(model, prep.X_test, prep.y_test)
        report = analyzer.full_report(sensitive_cols)

        # ── Metrics table ─────────────────────────────────────────────────────
        st.subheader("Metrics Table")
        st.dataframe(report, use_container_width=True)

        # ── Build readable group labels ───────────────────────────────────────
        group_labels: dict = {}
        for col in sensitive_cols:
            if col in prep.label_encoders:
                classes = prep.label_encoders[col].classes_
                group_labels[col] = {i: c for i, c in enumerate(classes)}
            elif col == "age_group":
                group_labels[col] = {0: "Young (<40)", 1: "Middle (40-60)", 2: "Senior (>60)"}
            else:
                group_labels[col] = {}

        # ── Grouped bar charts ────────────────────────────────────────────────
        st.subheader("Grouped Bar Charts")
        n = len(sensitive_cols)
        if n > 0:
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
            for i, col in enumerate(sensitive_cols):
                ax = axes[0][i]
                subset = report[report["sensitive_col"] == col].copy()
                g_map = group_labels.get(col, {})
                labels = [g_map.get(g, str(g)) for g in subset["group"]]
                x = np.arange(len(subset))
                w = 0.35
                ax.bar(x - w / 2, subset["demographic_parity"], w,
                       label="Dem. Parity", color="steelblue", alpha=0.85)
                ax.bar(x + w / 2, subset["equal_opportunity"], w,
                       label="Equal Opp.", color="coral", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
                ax.set_ylim(0, 1)
                ax.set_title(col, fontsize=11)
                ax.legend(fontsize=8)

            plt.tight_layout()

            # Save before rendering
            fairness_dir = Path(REPORTS_DIR) / "fairness" / "adult"
            fairness_dir.mkdir(parents=True, exist_ok=True)
            chart_path = fairness_dir / f"{model_name}_fairness.png"
            fig.savefig(str(chart_path), bbox_inches="tight", dpi=150)

            st.pyplot(fig)
            plt.close(fig)

        # ── Disparate Impact summary ──────────────────────────────────────────
        st.subheader("Disparate Impact Summary")
        di_summary = report.groupby("sensitive_col")["disparate_impact"].first()
        for col, di in di_summary.items():
            if di < 0.8:
                st.warning(
                    f"**{col}**: Disparate Impact = {di:.3f} — "
                    "below 0.8 (potential bias detected)"
                )
            else:
                st.success(
                    f"**{col}**: Disparate Impact = {di:.3f} — "
                    "above 0.8 (within acceptable range)"
                )

    except Exception as exc:
        st.error(f"Fairness analysis failed: {exc}")
