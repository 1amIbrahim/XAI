# Owner: Salman Ali Khan
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import pandas as pd

app = FastAPI(
    title="XAI Medical Decision Support API",
    description="Multi-model prediction with SHAP explanations and fairness analysis.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    dataset: Literal["heart_disease", "diabetes"]
    model: Literal["logistic_regression", "decision_tree", "random_forest", "neural_network"]
    features: dict


@app.get("/")
def root():
    return {"message": "XAI Medical Decision Support API is running."}


@app.post("/predict")
def predict(req: PredictRequest):
    # TODO (Salman): load saved model using ModelTrainer.load(), run prediction
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.post("/explain")
def explain(req: PredictRequest):
    # TODO (Salman): call SHAPExplainer, return shap values for the given instance
    raise HTTPException(status_code=501, detail="Not yet implemented")


@app.get("/fairness/{dataset}/{model}")
def fairness(dataset: str, model: str):
    # TODO (Salman): load model + test data, run FairnessAnalyzer.full_report()
    raise HTTPException(status_code=501, detail="Not yet implemented")
