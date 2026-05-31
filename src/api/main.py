# Owner: Salman Ali Khan
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import shap

from src.preprocessing.adult import AdultPreprocessor
from src.models.base import ModelTrainer
from src.fairness.metrics import FairnessAnalyzer
from config import ADULT_TARGET, ADULT_SENSITIVE_COLS

app = FastAPI(
    title="XAI Adult Income API",
    description="Multi-model prediction with SHAP explanations and fairness analysis.",
    version="1.0.0",
)

_preprocessors: dict = {}
_models: dict = {}


def _get_preprocessor(dataset: str) -> AdultPreprocessor:
    if dataset not in _preprocessors:
        if dataset == "adult":
            prep = AdultPreprocessor()
            prep.run(ADULT_TARGET)
            _preprocessors[dataset] = prep
        else:
            raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
    return _preprocessors[dataset]


def _get_model(dataset: str, model_name: str):
    key = f"{dataset}_{model_name}"
    if key not in _models:
        try:
            _models[key] = ModelTrainer.load(dataset, model_name)
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{dataset}/{model_name}' not found. Run train.py first.",
            )
    return _models[key]


class PredictRequest(BaseModel):
    dataset: Literal["adult"]
    model: Literal["logistic_regression", "decision_tree", "random_forest", "neural_network"]
    features: dict


@app.get("/")
def root():
    return {"message": "XAI Adult Income API is running."}


@app.post("/predict")
def predict(req: PredictRequest):
    prep = _get_preprocessor(req.dataset)
    model = _get_model(req.dataset, req.model)
    X = prep.transform_input(req.features)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return {"prediction": prediction, "probability": round(probability, 4)}


@app.post("/explain")
def explain(req: PredictRequest):
    prep = _get_preprocessor(req.dataset)
    model = _get_model(req.dataset, req.model)
    X = prep.transform_input(req.features)

    if req.model in ("random_forest", "decision_tree"):
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X)
    else:
        sample_size = min(200, len(prep.X_train))
        X_bg = prep.X_train.sample(n=sample_size, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, X_bg)
        raw = explainer.shap_values(X, nsamples=100)

    vals = np.array(raw[1] if isinstance(raw, list) else raw)
    if vals.ndim > 1:
        vals = vals[0]

    return {
        "shap_values": vals.tolist(),
        "feature_names": list(X.columns),
    }


@app.get("/fairness/{dataset}/{model}")
def fairness(dataset: str, model: str):
    valid_datasets = ["adult"]
    valid_models = ["logistic_regression", "decision_tree", "random_forest", "neural_network"]
    if dataset not in valid_datasets:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    prep = _get_preprocessor(dataset)
    loaded_model = _get_model(dataset, model)
    sensitive_cols = [c for c in ADULT_SENSITIVE_COLS if c in prep.X_test.columns]
    analyzer = FairnessAnalyzer(loaded_model, prep.X_test, prep.y_test)
    report = analyzer.full_report(sensitive_cols)
    return report.to_dict(orient="records")
