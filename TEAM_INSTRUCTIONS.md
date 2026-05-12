# Team Instructions — XAI System

**Project:** Multi-Model Explainable AI with SHAP + Fairness Analysis  
**Team:** Muhammad Ibrahim (BSAI23046) · Rabiya Tahir (BSAI23043) · Salman Ali Khan (BSAI23061)

---

## Datasets

| Who | Dataset | Location | Task |
|---|---|---|---|
| Rabiya | Heart Disease (UCI) | `data/heart_disease_data.csv` | Predict heart disease (0/1) |
| Salman | Adult Income (UCI Census) | `data/adult/adult.data` | Predict income >50K (0/1) |

Both datasets are **already in the repository**. No downloads needed.

---

## 1. One-Time Setup (Everyone)

```bash
# Clone the repo
git clone <repo-url>
cd XAI

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install all dependencies
pip install -r requirements.txt

# Create your feature branch
git checkout develop
git pull origin develop
git checkout -b feature/<your-branch-name>
```

**Branch names to use:**

| Work | Branch name |
|---|---|
| Rabiya — Phase 1 | `feature/rabiya-heart-disease` |
| Rabiya — Phase 2 | `feature/rabiya-fairness` |
| Salman — Phase 1 | `feature/salman-adult` |
| Salman — Phase 2 | `feature/salman-api-ui` |
| Ibrahim — Phase 2 | `feature/ibrahim-shap` |

---

## 2. Git Workflow (Follow Every Time)

```
main      ← final stable code only
  └── develop  ← everyone merges here via PR
        ├── feature/rabiya-heart-disease
        ├── feature/rabiya-fairness
        ├── feature/salman-adult
        ├── feature/salman-api-ui
        └── feature/ibrahim-shap
```

### Daily steps

```bash
# Before starting work — sync with develop
git checkout develop && git pull origin develop
git checkout feature/<your-branch>
git merge develop

# Work on your files, then commit
git add src/preprocessing/adult.py        # add specific files, never git add .
git commit -m "feat: implement adult income preprocessing"

# Push your branch
git push origin feature/<your-branch>

# Open a Pull Request to develop on GitHub
# Ibrahim reviews all PRs before merging
```

### Commit message format
```
feat: add adult income clean() method
fix: handle ? missing values in workclass column
docs: complete adult EDA notebook
```

### Rules
- **Never push directly to `main` or `develop`** — PR only
- **Only edit files in your assigned folders** — do not touch another person's files
- **Test your train script before opening a PR** — it must print accuracy for all 4 models

---

## 3. Phase 1 — ML Pipelines (Parallel Work)

### Muhammad Ibrahim — Repo Setup (COMPLETE)

Ibrahim has delivered:
- Full folder structure and `config.py`
- `src/preprocessing/base.py` — abstract base class
- `src/models/base.py` — `ModelTrainer` (trains + saves all 4 models)
- `requirements.txt`, notebooks, stubs for Rabiya and Salman

**Ibrahim's Phase 1 is done. He starts Phase 2 SHAP work after Rabiya and Salman deliver trained models.**

---

### Rabiya Tahir — Heart Disease Pipeline

**Branch:** `feature/rabiya-heart-disease`  
**Dataset:** `data/heart_disease_data.csv` (already in repo, no download needed)  
**Columns:** age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, **target**

#### Step 1 — EDA notebook
```bash
jupyter notebook notebooks/heart_disease_eda.ipynb
```
Answer the four TODO questions at the bottom before coding.

#### Step 2 — Implement `src/preprocessing/heart_disease.py`

Implement the three methods:

```python
def clean(self):
    self.df.dropna(inplace=True)              # or fill missing with median
    return self.df

def encode(self):
    # Heart Disease is mostly numeric, check df.dtypes first
    # Use pd.get_dummies() if any object columns remain
    return self.df

def scale(self):
    feature_cols = [c for c in self.df.columns if c != 'target']
    self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
    return self.df
```

#### Step 3 — Run the pipeline
```bash
python src/models/heart_disease/train.py
```

Expected output — 4 saved files:
```
saved_models/heart_disease_logistic_regression.pkl
saved_models/heart_disease_decision_tree.pkl
saved_models/heart_disease_random_forest.pkl
saved_models/heart_disease_neural_network.pkl
```

#### Step 4 — Open a PR to `develop`

---

### Salman Ali Khan — Adult Income Pipeline

**Branch:** `feature/salman-adult`  
**Dataset:** `data/adult/adult.data` (already in repo, no download needed)  
**Columns:** age, workclass, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, **income**  
**Note:** `fnlwgt` is automatically dropped in `load()` — you do not need to handle it.

#### Step 1 — EDA notebook
```bash
jupyter notebook notebooks/adult_eda.ipynb
```
Pay special attention to the Fairness Preview cell — it shows the income gap between Male/Female and across race groups that Rabiya will analyse in Phase 2.

#### Step 2 — Implement `src/preprocessing/adult.py`

```python
def clean(self):
    self.df.replace('?', pd.NA, inplace=True)
    self.df.dropna(inplace=True)                   # ~3k rows removed
    self.df['income'] = self.df['income'].str.strip()
    return self.df

def encode(self):
    for col in self.categorical_cols:
        le = LabelEncoder()
        self.df[col] = le.fit_transform(self.df[col].astype(str))
        self.label_encoders[col] = le
    # Binarise target: <=50K → 0, >50K → 1
    self.df['income'] = (self.df['income'] == '>50K').astype(int)
    return self.df

def scale(self):
    self.df[self.numerical_cols] = self.scaler.fit_transform(
        self.df[self.numerical_cols]
    )
    return self.df
```

#### Step 3 — Run the pipeline
```bash
python src/models/adult/train.py
```

Expected output — 4 saved files:
```
saved_models/adult_logistic_regression.pkl
saved_models/adult_decision_tree.pkl
saved_models/adult_random_forest.pkl
saved_models/adult_neural_network.pkl
```

Typical accuracy on Adult dataset: ~83–87% depending on model.

#### Step 4 — Open a PR to `develop`

---

## 4. Phase 2 — Specialization (After Phase 1 PRs are merged)

### Muhammad Ibrahim — SHAP Explainability

**Branch:** `feature/ibrahim-shap`  
**File:** `src/explainability/shap_explainer.py` (scaffold already written)

You need `X_train` and `X_test` from both datasets — re-run the preprocessors locally to get them.

**Deliverables** (saved to `reports/shap/<dataset>/`):
- Summary plot per model × per dataset — global feature importance
- Waterfall plot for one sample — local explanation (why did this person get this prediction?)
- Dependence plot for top 2 features per dataset

**Usage:**
```python
from src.explainability.shap_explainer import SHAPExplainer

# Run all plots for Random Forest on Heart Disease
explainer = SHAPExplainer("heart_disease", "random_forest", X_train)
explainer.run_all(X_test, top_feature="thalach")

# Run for Adult Income
explainer = SHAPExplainer("adult", "random_forest", X_train)
explainer.run_all(X_test, top_feature="age")
```

**Adult dataset SHAP note:** Use `TreeExplainer` for Random Forest and Decision Tree (fast). For Logistic Regression and Neural Network, `KernelExplainer` is used automatically but is slow — use a sample of 200 rows for `X_test` when testing.

---

### Rabiya Tahir — Fairness Analysis

**Branch:** `feature/rabiya-fairness`  
**File:** `src/fairness/metrics.py` (scaffold already written)

Implement the 4 methods in `FairnessAnalyzer`:

| Method | What it computes |
|---|---|
| `demographic_parity(col)` | Positive prediction rate per group |
| `equal_opportunity(col)` | True positive rate per group |
| `disparate_impact(col)` | min_rate / max_rate — below 0.8 indicates bias |
| `full_report(cols)` | Summary DataFrame for all metrics × all sensitive columns |

**Sensitive columns to analyze:**

| Dataset | Sensitive columns | Notes |
|---|---|---|
| Heart Disease | `sex`, `age_group` | Bin `age` into Young (<40), Middle (40–60), Senior (>60) |
| Adult Income | `sex`, `race`, `age_group` | Same age binning; race has 5 categories |

**Expected findings from literature:** On the Adult dataset, males are predicted >50K at roughly 3× the rate of females. Your analysis should quantify and visualise this.

Save fairness charts to `reports/fairness/<dataset>/`.

---

### Salman Ali Khan — API + Dashboard

**Branch:** `feature/salman-api-ui`  
**Files:** `src/api/main.py`, `dashboard/app.py`

#### FastAPI — implement 3 endpoints

```python
# POST /predict
# Body: {"dataset": "adult", "model": "random_forest", "features": {...}}
# Returns: {"prediction": 1, "probability": 0.83}

# POST /explain
# Body: same as /predict
# Returns: {"shap_values": [...], "feature_names": [...]}

# GET /fairness/{dataset}/{model}
# Returns: fairness report as JSON
```

Test with the auto-generated docs:
```bash
uvicorn src.api.main:app --reload
# Open: http://127.0.0.1:8000/docs
```

#### Streamlit — 3 pages in `dashboard/app.py`

```python
page = st.sidebar.selectbox("Page", ["Prediction", "Explanation", "Fairness"])
```

1. **Prediction** — dropdowns for dataset + model, input fields for features, show prediction + confidence bar
2. **Explanation** — display SHAP summary and waterfall plots from `reports/shap/`
3. **Fairness** — show fairness metrics table and grouped bar charts from `reports/fairness/`

```bash
streamlit run dashboard/app.py
```

---

## 5. Phase 3 — Integration (Everyone)

1. Salman runs full system end-to-end; fixes any connection issues between API and dashboard
2. Ibrahim confirms SHAP plots display correctly on the Explanation page
3. Rabiya confirms fairness metrics and charts render correctly on the Fairness page
4. Final PR from `develop` → `main`

---

## 6. Model Naming Convention

**Critical — do not deviate. Phase 2 depends on this.**

```
saved_models/<dataset_name>_<model_name>.pkl
```

Valid values:
- `dataset_name`: `heart_disease`, `adult`
- `model_name`: `logistic_regression`, `decision_tree`, `random_forest`, `neural_network`

Load from anywhere:
```python
from src.models.base import ModelTrainer
model = ModelTrainer.load("adult", "random_forest")
```

---

## 7. Common Issues

**`ModuleNotFoundError: No module named 'config'`**  
Always run from the project root:
```bash
cd XAI
python src/models/adult/train.py    # correct
# NOT: cd src/models/adult && python train.py
```

**`FileNotFoundError: saved_models/adult_random_forest.pkl`**  
Phase 2 needs Phase 1 done first. Run the train scripts to generate the `.pkl` files.

**`NotImplementedError: Salman: implement clean()`**  
The scaffold raises this until you fill in the method body. Remove the `raise` line and write the implementation.

**Merge conflict**  
File ownership is split so conflicts are rare. If one occurs, keep both changes and test. Tag Ibrahim on the PR for help.
