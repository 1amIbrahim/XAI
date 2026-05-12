# XAI Medical Decision Support System

A Multi-Model Explainable AI system using SHAP with Fairness Analysis.  
Predicts outcomes on two healthcare/socioeconomic datasets and explains model decisions.

## Team

| Name | Role | Branch |
|---|---|---|
| Muhammad Ibrahim (BSAI23046) | SHAP & Explainability | `feature/ibrahim-shap` |
| Rabiya Tahir (BSAI23043) | Heart Disease Pipeline + Fairness Analysis | `feature/rabiya-heart-disease`, `feature/rabiya-fairness` |
| Salman Ali Khan (BSAI23061) | Adult Income Pipeline + API + Dashboard | `feature/salman-adult`, `feature/salman-api-ui` |

## Datasets

| Dataset | File | Task | Rows | Features |
|---|---|---|---|---|
| Heart Disease (UCI) | `data/heart_disease_data.csv` | Predict heart disease (0/1) | ~303 | 14 |
| Adult Income (UCI Census) | `data/adult/adult.data` | Predict income >50K (0/1) | 32,561 | 14 |

Both datasets are already in the repository.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd XAI
pip install -r requirements.txt

# 2. Run Heart Disease pipeline (Rabiya — Phase 1)
python src/models/heart_disease/train.py

# 3. Run Adult Income pipeline (Salman — Phase 1)
python src/models/adult/train.py

# 4. Run FastAPI backend (Salman — Phase 2)
uvicorn src.api.main:app --reload

# 5. Run Streamlit dashboard (Salman — Phase 2)
streamlit run dashboard/app.py
```

> Always run scripts from the project root (`cd XAI`), not from inside subfolders.

## Project Structure

```
XAI/
├── config.py                       # Central paths and constants — import from here
├── requirements.txt
├── data/
│   ├── heart_disease_data.csv      # Rabiya's dataset
│   ├── adult/
│   │   ├── adult.data              # Salman's dataset (training set, 32k rows)
│   │   └── adult.names             # Column descriptions
│   └── processed/                  # Auto-generated (gitignored)
├── notebooks/
│   ├── heart_disease_eda.ipynb     # Rabiya's EDA
│   └── adult_eda.ipynb             # Salman's EDA
├── src/
│   ├── preprocessing/
│   │   ├── base.py                 # Abstract BasePreprocessor (Ibrahim — done)
│   │   ├── heart_disease.py        # Rabiya implements
│   │   └── adult.py                # Salman implements
│   ├── models/
│   │   ├── base.py                 # ModelTrainer — trains + saves all 4 models (Ibrahim — done)
│   │   ├── heart_disease/
│   │   │   └── train.py            # Rabiya runs
│   │   └── adult/
│   │       └── train.py            # Salman runs
│   ├── explainability/
│   │   └── shap_explainer.py       # Ibrahim — Phase 2
│   ├── fairness/
│   │   └── metrics.py              # Rabiya — Phase 2
│   └── api/
│       └── main.py                 # Salman — Phase 2
├── dashboard/
│   └── app.py                      # Salman — Phase 2
├── saved_models/                   # Auto-generated .pkl files (gitignored)
└── reports/                        # Auto-generated plots (gitignored)
    ├── shap/
    └── fairness/
```

## Model Naming Convention

```
saved_models/<dataset_name>_<model_name>.pkl
```

| dataset_name | model_name options |
|---|---|
| `heart_disease` | `logistic_regression` `decision_tree` `random_forest` `neural_network` |
| `adult` | `logistic_regression` `decision_tree` `random_forest` `neural_network` |

Load anywhere with:
```python
from src.models.base import ModelTrainer
model = ModelTrainer.load("adult", "random_forest")
```

## Why the Adult Dataset?

The Adult Income dataset (UCI Census, 1994) has explicit demographic columns — `sex`, `race`, `age`, `native-country` — making fairness analysis concrete and meaningful. Historical studies have confirmed real bias in income prediction models on this dataset, so the SHAP + fairness analysis will surface genuine, interpretable findings.

See `TEAM_INSTRUCTIONS.md` for the full workflow.
