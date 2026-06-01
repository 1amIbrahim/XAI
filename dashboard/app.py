# Owner: Salman Ali Khan
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import shap

from config import ADULT_TARGET, ADULT_SENSITIVE_COLS, REPORTS_DIR

st.set_page_config(
    page_title="XAI — Income Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Demo persona cards */
.demo-card {
    background: linear-gradient(145deg, rgba(99,102,241,0.07), rgba(139,92,246,0.07));
    border: 1.5px solid rgba(99,102,241,0.22);
    border-radius: 14px;
    padding: 16px 10px 12px;
    text-align: center;
    margin-bottom: 4px;
    min-height: 148px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 3px;
    transition: border-color 0.2s, transform 0.15s;
}
.demo-card:hover { border-color: rgba(99,102,241,0.55); transform: translateY(-2px); }
.demo-emoji  { font-size: 2.1rem; line-height: 1; }
.demo-name   { font-weight: 700; font-size: 0.88rem; margin-top: 4px; }
.demo-desc   { font-size: 0.73rem; color: #888; }

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    margin-top: 6px;
    letter-spacing: 0.02em;
}
.badge-green  { background:rgba(16,185,129,0.12); color:#10b981; border:1px solid rgba(16,185,129,0.35); }
.badge-orange { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.35); }
.badge-blue   { background:rgba(59,130,246,0.12); color:#3b82f6; border:1px solid rgba(59,130,246,0.35); }
.badge-purple { background:rgba(139,92,246,0.12); color:#8b5cf6; border:1px solid rgba(139,92,246,0.35); }

/* Result card */
.result-card {
    border-radius: 14px;
    padding: 20px 24px;
    margin: 6px 0;
}
.result-positive { background:rgba(16,185,129,0.10); border:1.5px solid rgba(16,185,129,0.35); }
.result-negative { background:rgba(245,158,11,0.10); border:1.5px solid rgba(245,158,11,0.35); }
.result-label { font-size: 1.5rem; font-weight: 800; margin-bottom: 4px; }
.result-prob  { font-size: 0.9rem; color: #888; }

/* Section label */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin: 14px 0 6px;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] * { color: #e0e0f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Category constants ────────────────────────────────────────────────────────
EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}
WORKCLASS   = ["Federal-gov", "Local-gov", "Never-worked", "Private",
               "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"]
EDUCATION   = list(EDUCATION_NUM.keys())
MARITAL     = ["Divorced", "Married-AF-spouse", "Married-civ-spouse",
               "Married-spouse-absent", "Never-married", "Separated", "Widowed"]
OCCUPATION  = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
               "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
               "Other-service", "Priv-house-serv", "Prof-specialty",
               "Protective-serv", "Sales", "Tech-support", "Transport-moving"]
RELATIONSHIP = ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
RACE        = ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]
SEX         = ["Female", "Male"]
COUNTRY     = [
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
    "decision_tree":       "Decision Tree",
    "random_forest":       "Random Forest",
    "neural_network":      "Neural Network (MLP)",
}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}

# ── Demo personas ─────────────────────────────────────────────────────────────
# Each persona is designed to highlight a different aspect of the model or bias.
DEMO_PERSONAS = {
    "exec": {
        "emoji": "👨‍💼", "name": "Tech Executive",
        "desc": "Male · 42 · Exec-managerial",
        "badge": ">50K likely", "badge_color": "green",
        "features": {
            "age": 42, "workclass": "Private", "education": "Bachelors",
            "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 7688, "capital-loss": 0,
            "hours-per-week": 50, "native-country": "United-States",
        },
    },
    "grad": {
        "emoji": "👩‍🎓", "name": "Young Graduate",
        "desc": "Female · 27 · Prof-specialty",
        "badge": "<=50K likely", "badge_color": "orange",
        "features": {
            "age": 27, "workclass": "Private", "education": "Bachelors",
            "marital-status": "Never-married", "occupation": "Prof-specialty",
            "relationship": "Not-in-family", "race": "White", "sex": "Female",
            "capital-gain": 0, "capital-loss": 0,
            "hours-per-week": 40, "native-country": "United-States",
        },
    },
    "doctor": {
        "emoji": "👩‍⚕️", "name": "Senior Doctor",
        "desc": "Female · 45 · Doctorate · Fairness test",
        "badge": "Fairness test", "badge_color": "purple",
        "features": {
            "age": 45, "workclass": "Private", "education": "Doctorate",
            "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
            "relationship": "Wife", "race": "White", "sex": "Female",
            "capital-gain": 0, "capital-loss": 0,
            "hours-per-week": 50, "native-country": "United-States",
        },
    },
    "farmer": {
        "emoji": "🌾", "name": "Farm Worker",
        "desc": "Male · 55 · HS-grad · Self-employed",
        "badge": "<=50K likely", "badge_color": "orange",
        "features": {
            "age": 55, "workclass": "Self-emp-not-inc", "education": "HS-grad",
            "marital-status": "Married-civ-spouse", "occupation": "Farming-fishing",
            "relationship": "Husband", "race": "White", "sex": "Male",
            "capital-gain": 0, "capital-loss": 0,
            "hours-per-week": 60, "native-country": "United-States",
        },
    },
    "gov": {
        "emoji": "🏛️", "name": "Gov. Officer",
        "desc": "Female · 38 · Masters · Federal",
        "badge": "Borderline", "badge_color": "blue",
        "features": {
            "age": 38, "workclass": "Federal-gov", "education": "Masters",
            "marital-status": "Never-married", "occupation": "Prof-specialty",
            "relationship": "Unmarried", "race": "Black", "sex": "Female",
            "capital-gain": 0, "capital-loss": 0,
            "hours-per-week": 40, "native-country": "United-States",
        },
    },
}

# Maps persona feature keys (with dashes) → session state keys (with underscores)
_SS_KEY = {
    "age":            "feat_age",
    "workclass":      "feat_workclass",
    "education":      "feat_education",
    "marital-status": "feat_marital",
    "occupation":     "feat_occupation",
    "relationship":   "feat_relationship",
    "race":           "feat_race",
    "sex":            "feat_sex",
    "capital-gain":   "feat_cap_gain",
    "capital-loss":   "feat_cap_loss",
    "hours-per-week": "feat_hours",
    "native-country": "feat_country",
}

_DEFAULTS = {
    "feat_age":        35,
    "feat_workclass":  "Private",
    "feat_education":  "Bachelors",
    "feat_marital":    "Never-married",
    "feat_occupation": "Prof-specialty",
    "feat_relationship": "Not-in-family",
    "feat_race":       "White",
    "feat_sex":        "Male",
    "feat_cap_gain":   0,
    "feat_cap_loss":   0,
    "feat_hours":      40,
    "feat_country":    "United-States",
}

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


# ── Live SHAP helper ──────────────────────────────────────────────────────────

def _live_shap_waterfall(model, model_name: str, X: pd.DataFrame, X_train: pd.DataFrame):
    """Compute SHAP values for one input row → return (figure, base_value)."""
    feature_names = list(X.columns)
    n_features = len(feature_names)

    if model_name in ("random_forest", "decision_tree"):
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
    elif model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_train)
        raw = explainer.shap_values(X)
    else:
        background = X_train.sample(min(50, len(X_train)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(X, nsamples=50, silent=True)

    raw_arr = np.array(raw)
    if raw_arr.ndim == 3:
        if raw_arr.shape == (1, n_features, 2):
            shap_vals = raw_arr[0, :, 1]
        elif raw_arr.shape == (2, 1, n_features):
            shap_vals = raw_arr[1, 0, :]
        else:
            shap_vals = raw_arr.reshape(-1, n_features)[0]
        ev = explainer.expected_value
        base_val = float(ev[1]) if hasattr(ev, "__len__") else float(ev)
    else:
        shap_vals = raw_arr[0]
        ev = explainer.expected_value
        base_val = float(ev) if np.isscalar(ev) else float(ev[0])

    exp = shap.Explanation(
        values=shap_vals,
        base_values=base_val,
        data=X.iloc[0].values,
        feature_names=feature_names,
    )
    shap.plots.waterfall(exp, show=False)
    return plt.gcf(), base_val


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 XAI Dashboard")
st.sidebar.markdown("*Adult Income · UCI Census 1994*")
st.sidebar.divider()
page = st.sidebar.selectbox("Navigate", ["Prediction", "Explanation", "Fairness"],
                             label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("""
**Pages**
- **Prediction** — run a model on a person's profile
- **Explanation** — SHAP plots showing what drives decisions
- **Fairness** — bias audit across sex, race, and age
""")
st.sidebar.divider()
st.sidebar.caption("Department of AI · Semester 6 · 2026")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION PAGE
# ─────────────────────────────────────────────────────────────────────────────
if page == "Prediction":

    st.markdown("# Income Prediction")
    st.markdown(
        "Predict whether an individual's annual income exceeds **\\$50K**, "
        "compare all four models, explore what-if changes, and get an instant SHAP explanation."
    )

    # ── Initialize session state defaults ────────────────────────────────────
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Demo persona cards ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Quick demo profiles — click to load</p>',
                unsafe_allow_html=True)
    demo_cols = st.columns(5, gap="small")
    for col, (pid, p) in zip(demo_cols, DEMO_PERSONAS.items()):
        with col:
            st.markdown(f"""
            <div class="demo-card">
                <div class="demo-emoji">{p['emoji']}</div>
                <div class="demo-name">{p['name']}</div>
                <div class="demo-desc">{p['desc']}</div>
                <span class="badge badge-{p['badge_color']}">{p['badge']}</span>
            </div>""", unsafe_allow_html=True)
            if st.button("Load", key=f"demo_{pid}", use_container_width=True):
                for field, val in p["features"].items():
                    st.session_state[_SS_KEY[field]] = val
                # Clear previous results when a new profile is loaded
                st.session_state.pop("pred_result", None)
                st.rerun()

    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    model_label = st.selectbox("Model", list(MODEL_LABELS.values()),
                               label_visibility="collapsed")
    model_name = LABEL_TO_KEY[model_label]

    # ── Feature form ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Individual features</p>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.number_input("Age", min_value=17, max_value=90, key="feat_age")
        st.selectbox("Workclass", WORKCLASS, key="feat_workclass")
        st.selectbox("Education", EDUCATION, key="feat_education")
        st.number_input("Capital Gain ($)", min_value=0, max_value=99999, key="feat_cap_gain")

    with c2:
        st.selectbox("Marital Status", MARITAL, key="feat_marital")
        st.selectbox("Occupation", OCCUPATION, key="feat_occupation")
        st.selectbox("Relationship", RELATIONSHIP, key="feat_relationship")
        st.number_input("Capital Loss ($)", min_value=0, max_value=99999, key="feat_cap_loss")

    with c3:
        st.selectbox("Race", RACE, key="feat_race")
        st.selectbox("Sex", SEX, key="feat_sex")
        st.slider("Hours per Week", min_value=1, max_value=99, key="feat_hours")
        st.selectbox("Native Country", COUNTRY, key="feat_country")

    st.markdown("")
    predict_btn = st.button("Predict Income", type="primary", use_container_width=True)

    # ── Run prediction and store results ─────────────────────────────────────
    if predict_btn:
        features = {
            "age":            st.session_state["feat_age"],
            "workclass":      st.session_state["feat_workclass"],
            "education":      st.session_state["feat_education"],
            "education-num":  EDUCATION_NUM[st.session_state["feat_education"]],
            "marital-status": st.session_state["feat_marital"],
            "occupation":     st.session_state["feat_occupation"],
            "relationship":   st.session_state["feat_relationship"],
            "race":           st.session_state["feat_race"],
            "sex":            st.session_state["feat_sex"],
            "capital-gain":   st.session_state["feat_cap_gain"],
            "capital-loss":   st.session_state["feat_cap_loss"],
            "hours-per-week": st.session_state["feat_hours"],
            "native-country": st.session_state["feat_country"],
        }
        try:
            prep  = load_preprocessor()
            model = load_model(model_name)
            X     = prep.transform_input(features)
            prediction  = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            st.session_state["pred_result"] = {
                "prediction":  prediction,
                "probability": probability,
                "features":    features,
                "model_name":  model_name,
                "model_label": model_label,
            }
            # Seed what-if controls from the current features
            st.session_state["wif_age"]        = features["age"]
            st.session_state["wif_education"]  = features["education"]
            st.session_state["wif_sex"]        = features["sex"]
            st.session_state["wif_marital"]    = features["marital-status"]
            st.session_state["wif_hours"]      = features["hours-per-week"]
            st.session_state["wif_cap_gain"]   = features["capital-gain"]
            st.session_state["wif_occupation"] = features["occupation"]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    # ── Results section (persists across what-if re-runs) ────────────────────
    if "pred_result" in st.session_state:
        r           = st.session_state["pred_result"]
        prediction  = r["prediction"]
        probability = r["probability"]
        features    = r["features"]
        model_name  = r["model_name"]
        model_label = r["model_label"]
        label       = ">50K" if prediction == 1 else "≤50K"
        card_cls    = "result-positive" if prediction == 1 else "result-negative"
        icon        = "✅" if prediction == 1 else "⚠️"

        prep  = load_preprocessor()
        model = load_model(model_name)
        X     = prep.transform_input(features)

        st.divider()

        # ── Result card ───────────────────────────────────────────────────────
        rc, pc, sc = st.columns([1.4, 1.8, 1], gap="medium")
        with rc:
            st.markdown(f"""
            <div class="result-card {card_cls}">
                <div class="result-label">{icon} {label}</div>
                <div class="result-prob">Predicted income class</div>
            </div>""", unsafe_allow_html=True)
        with pc:
            st.metric("Probability of income > $50K", f"{probability:.1%}",
                      delta=f"{probability - 0.5:+.1%} vs 50/50")
            st.progress(probability)
        with sc:
            st.metric("Model", model_label.replace(" (MLP)", ""))
            st.metric("Education", features["education"])

        # ─────────────────────────────────────────────────────────────────────
        # MODEL COMPARISON
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Model Comparison")
        st.caption("Same person · same features · all four models. "
                   "Agreement = confidence. Disagreement = uncertainty worth investigating.")

        comp_probs, comp_preds = {}, {}
        for mn, ml in MODEL_LABELS.items():
            try:
                m = load_model(mn)
                p = float(m.predict_proba(X)[0][1])
                comp_probs[ml] = p
                comp_preds[ml] = int(m.predict(X)[0])
            except Exception:
                comp_probs[ml] = None

        # Horizontal bar chart
        valid = {k: v for k, v in comp_probs.items() if v is not None}
        names  = list(valid.keys())
        probs  = list(valid.values())
        colors = ["#10b981" if p >= 0.6 else "#ef4444" if p <= 0.4 else "#f59e0b"
                  for p in probs]

        fig_cmp, ax_cmp = plt.subplots(figsize=(8, 2.8))
        bars = ax_cmp.barh(names, probs, color=colors, alpha=0.88, height=0.45)
        ax_cmp.axvline(0.5, color="#888", linestyle="--", linewidth=1.2, alpha=0.6,
                       label="Decision boundary (0.5)")
        ax_cmp.set_xlim(0, 1.12)
        ax_cmp.set_xlabel("P(income > $50K)", fontsize=9)
        ax_cmp.set_title("All-Model Probability Comparison", fontweight="bold", fontsize=11)
        for bar, prob in zip(bars, probs):
            clr = "#10b981" if prob >= 0.6 else "#ef4444" if prob <= 0.4 else "#f59e0b"
            ax_cmp.text(prob + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1%}", va="center", fontsize=9.5,
                        color=clr, fontweight="bold")
        ax_cmp.legend(fontsize=8, loc="lower right")
        ax_cmp.spines[["top", "right"]].set_visible(False)
        ax_cmp.tick_params(axis="y", labelsize=9)
        plt.tight_layout()
        st.pyplot(fig_cmp, use_container_width=True)
        plt.close(fig_cmp)

        # Agreement summary
        votes = list(comp_preds.values())
        n_pos = sum(votes)
        n_total = len(votes)
        if n_pos == n_total:
            st.success(f"**All {n_total} models agree:** income > $50K", icon="✅")
        elif n_pos == 0:
            st.warning(f"**All {n_total} models agree:** income ≤ $50K", icon="⚠️")
        else:
            st.info(f"**Models split {n_pos}–{n_total - n_pos}** — prediction is uncertain near the boundary.", icon="🤔")

        # ─────────────────────────────────────────────────────────────────────
        # WHAT-IF EXPLORER
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### What-If Explorer")
        st.caption(
            "Adjust the sliders and dropdowns below to see how changes affect the prediction. "
            "All other features stay fixed. Use this to find what would flip the outcome."
        )

        # Initialize what-if keys if missing
        wif_defaults = {
            "wif_age":        features["age"],
            "wif_education":  features["education"],
            "wif_sex":        features["sex"],
            "wif_marital":    features["marital-status"],
            "wif_hours":      features["hours-per-week"],
            "wif_cap_gain":   features["capital-gain"],
            "wif_occupation": features["occupation"],
        }
        for k, v in wif_defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        wif_left, wif_right = st.columns([1.6, 1], gap="large")

        with wif_left:
            st.markdown('<p class="section-label">Adjust features</p>',
                        unsafe_allow_html=True)
            r1a, r1b = st.columns(2)
            with r1a:
                st.selectbox("Education", EDUCATION, key="wif_education")
                st.slider("Age", min_value=17, max_value=90, key="wif_age")
                st.number_input("Capital Gain ($)", min_value=0,
                                max_value=99999, key="wif_cap_gain")
            with r1b:
                st.radio("Sex", SEX, key="wif_sex", horizontal=True)
                st.selectbox("Marital Status", MARITAL, key="wif_marital")
                st.slider("Hours / Week", min_value=1, max_value=99, key="wif_hours")
            st.selectbox("Occupation", OCCUPATION, key="wif_occupation")

        # Build what-if features (override 7 fields, keep rest from original)
        wif_features = dict(features)
        wif_features["age"]            = st.session_state["wif_age"]
        wif_features["education"]      = st.session_state["wif_education"]
        wif_features["education-num"]  = EDUCATION_NUM[st.session_state["wif_education"]]
        wif_features["sex"]            = st.session_state["wif_sex"]
        wif_features["marital-status"] = st.session_state["wif_marital"]
        wif_features["hours-per-week"] = st.session_state["wif_hours"]
        wif_features["capital-gain"]   = st.session_state["wif_cap_gain"]
        wif_features["occupation"]     = st.session_state["wif_occupation"]

        try:
            X_wif        = prep.transform_input(wif_features)
            wif_pred     = int(model.predict(X_wif)[0])
            wif_prob     = float(model.predict_proba(X_wif)[0][1])
            delta_prob   = wif_prob - probability
            flipped      = wif_pred != prediction

            with wif_right:
                st.markdown('<p class="section-label">What-If result</p>',
                            unsafe_allow_html=True)

                if flipped:
                    new_label = ">50K" if wif_pred == 1 else "≤50K"
                    st.error(f"⚡ Prediction flipped to **{new_label}**!", icon="⚡")

                wif_cls = "result-positive" if wif_pred == 1 else "result-negative"
                wif_icon = "✅" if wif_pred == 1 else "⚠️"
                st.markdown(f"""
                <div class="result-card {wif_cls}">
                    <div class="result-label">{wif_icon} {">50K" if wif_pred == 1 else "≤50K"}</div>
                    <div class="result-prob">What-If prediction</div>
                </div>""", unsafe_allow_html=True)

                sign = "+" if delta_prob >= 0 else ""
                delta_color = "#10b981" if delta_prob >= 0 else "#ef4444"
                st.markdown(f"""
                <div style="margin-top:12px;">
                    <div style="font-size:2rem;font-weight:800;">{wif_prob:.1%}</div>
                    <div style="font-size:0.9rem;color:{delta_color};font-weight:600;">
                        {sign}{delta_prob:.1%} vs original
                    </div>
                    <div style="font-size:0.75rem;color:#888;margin-top:2px;">
                        Original: {probability:.1%}
                    </div>
                </div>""", unsafe_allow_html=True)

                st.progress(wif_prob)

                st.markdown("")
                if st.button("Run Full Predict + SHAP on this What-If",
                             use_container_width=True):
                    for field, ss_key in _SS_KEY.items():
                        wif_key = {
                            "age": "wif_age", "education": "wif_education",
                            "sex": "wif_sex", "marital-status": "wif_marital",
                            "hours-per-week": "wif_hours", "capital-gain": "wif_cap_gain",
                            "occupation": "wif_occupation",
                        }.get(field)
                        if wif_key:
                            st.session_state[ss_key] = st.session_state[wif_key]
                    st.session_state.pop("pred_result", None)
                    st.rerun()

        except Exception as wif_err:
            with wif_right:
                st.warning(f"What-If compute error: {wif_err}")

        # ─────────────────────────────────────────────────────────────────────
        # SHAP EXPLANATION
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Why this prediction?")
        st.caption(
            "SHAP values for the **original** prediction. "
            "Red = pushed toward >50K · Blue = pushed toward ≤50K"
        )
        if model_name == "neural_network":
            st.info("Neural Network explanations use KernelExplainer — takes ~1 minute.", icon="⏳")

        with st.spinner("Computing SHAP explanation…"):
            try:
                shap_fig, base_val = _live_shap_waterfall(model, model_name, X, prep.X_train)
                st.pyplot(shap_fig, use_container_width=True)
                plt.close("all")
                st.caption(
                    f"Base value **{base_val:.1%}** is the model's average prediction. "
                    f"The bars shift it to the final **{probability:.1%}**."
                )
            except Exception as shap_err:
                st.warning(f"SHAP explanation unavailable: {shap_err}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Explanation":
    st.markdown("# SHAP Explanations")
    st.markdown(
        "Pre-computed SHAP plots across the full test set. "
        "**Summary** shows global feature importance · **Waterfall** shows one sample's local explanation."
    )

    model_label = st.selectbox("Model", list(MODEL_LABELS.values()))
    model_name  = LABEL_TO_KEY[model_label]

    shap_dir      = Path(REPORTS_DIR) / "shap" / "adult"
    summary_path  = shap_dir / f"{model_name}_summary.png"
    waterfall_path = shap_dir / f"{model_name}_waterfall.png"
    bar_path      = shap_dir / f"{model_name}_bar.png"

    if not shap_dir.exists() or not summary_path.exists():
        st.info(
            "SHAP plots not found. Run `python src/explainability/run_shap.py` first.",
            icon="ℹ️",
        )
    else:
        tab1, tab2, tab3 = st.tabs(["Beeswarm Summary", "Feature Importance Bar", "Waterfall"])

        with tab1:
            st.markdown("Each dot is one test sample. **X-axis** = SHAP value. "
                        "**Colour** = feature value (red = high, blue = low).")
            if summary_path.exists():
                st.image(str(summary_path), use_container_width=True)

        with tab2:
            st.markdown("Mean absolute SHAP value per feature — clean global ranking.")
            if bar_path.exists():
                st.image(str(bar_path), use_container_width=True)
            else:
                st.warning("Bar plot not found.")

        with tab3:
            st.markdown("Local explanation for one test sample — why did that prediction happen?")
            if waterfall_path.exists():
                st.image(str(waterfall_path), use_container_width=True)
            else:
                st.warning("Waterfall plot not found. Re-run `run_shap.py`.")


# ─────────────────────────────────────────────────────────────────────────────
# FAIRNESS PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Fairness":
    st.markdown("# Fairness Analysis")
    st.markdown(
        "Measures whether the model treats demographic groups equitably. "
        "Sensitive attributes: **sex**, **race**, **age group**."
    )

    model_label = st.selectbox("Model", list(MODEL_LABELS.values()))
    model_name  = LABEL_TO_KEY[model_label]

    # ── Metric definitions ────────────────────────────────────────────────────
    with st.expander("What do these metrics mean?", expanded=False):
        st.markdown("""
| Metric | Definition | Fair if… |
|---|---|---|
| **Demographic Parity** | Fraction of each group predicted as >50K | Rates are equal across groups |
| **Equal Opportunity** | True positive rate (recall) per group | TPR is equal across groups |
| **Disparate Impact** | Lowest rate ÷ highest rate across groups | Value ≥ 0.8 (80% rule) |
        """)

    try:
        prep  = load_preprocessor()
        model = load_model(model_name)
        from src.fairness.metrics import FairnessAnalyzer

        sensitive_cols = [c for c in ADULT_SENSITIVE_COLS if c in prep.X_test.columns]
        analyzer = FairnessAnalyzer(model, prep.X_test, prep.y_test)
        report   = analyzer.full_report(sensitive_cols)

        # ── Disparate Impact alerts (top of page) ─────────────────────────────
        st.markdown('<p class="section-label">Disparate Impact Alerts</p>',
                    unsafe_allow_html=True)
        di_summary = report.groupby("sensitive_col")["disparate_impact"].first()
        alert_cols = st.columns(len(di_summary))
        for col, (attr, di) in zip(alert_cols, di_summary.items()):
            with col:
                if di < 0.8:
                    st.error(f"**{attr}**\n\nDI = {di:.3f}\n\n⚠️ Bias detected", icon="🚨")
                else:
                    st.success(f"**{attr}**\n\nDI = {di:.3f}\n\n✅ Within threshold", icon="✅")

        st.divider()

        # ── Build readable group labels ───────────────────────────────────────
        group_labels: dict = {}
        for col in sensitive_cols:
            if col in prep.label_encoders:
                classes = prep.label_encoders[col].classes_
                group_labels[col] = {i: c for i, c in enumerate(classes)}
            elif col == "age_group":
                group_labels[col] = {0: "Young (<40)", 1: "Middle (40–60)", 2: "Senior (>60)"}

        # ── Bar charts + table as tabs ────────────────────────────────────────
        tab_chart, tab_table = st.tabs(["Bar Charts", "Full Metrics Table"])

        with tab_chart:
            n = len(sensitive_cols)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
            for i, col in enumerate(sensitive_cols):
                ax     = axes[0][i]
                subset = report[report["sensitive_col"] == col].copy()
                g_map  = group_labels.get(col, {})
                labels = [g_map.get(g, str(g)) for g in subset["group"]]
                x, w   = np.arange(len(subset)), 0.35
                ax.bar(x - w / 2, subset["demographic_parity"], w,
                       label="Dem. Parity",    color="#6366f1", alpha=0.85)
                ax.bar(x + w / 2, subset["equal_opportunity"],  w,
                       label="Equal Opp.",     color="#f472b6", alpha=0.85)
                ax.axhline(0.8, color="red", linewidth=1, linestyle="--",
                           alpha=0.5, label="0.8 threshold")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
                ax.set_ylim(0, 1.05)
                ax.set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
                ax.legend(fontsize=7)
                ax.spines[["top", "right"]].set_visible(False)

            plt.tight_layout()
            fairness_dir = Path(REPORTS_DIR) / "fairness" / "adult"
            fairness_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fairness_dir / f"{model_name}_fairness.png"),
                        bbox_inches="tight", dpi=150)
            st.pyplot(fig)
            plt.close(fig)

        with tab_table:
            st.dataframe(report, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Fairness analysis failed: {exc}")
