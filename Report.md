# Explainable AI Decision Support System

## Professional Project Report

**Project:** Multi-Model Explainable AI with SHAP, Fairness Analysis, and Interactive Dashboard  
**Team:** Muhammad Ibrahim, Rabiya Tahir, Salman Ali Khan  
**Course:** Semester 6 Machine Learning Project  

---

## 1. Executive Summary

Machine learning models are increasingly used in high-stakes decision-making, including healthcare risk prediction and socioeconomic classification. In these settings, predictive accuracy alone is not enough. A model must also provide understandable explanations and must be audited for unfair behavior across demographic groups.

This project presents an Explainable AI decision-support system that combines three capabilities:

1. Multi-model prediction using Logistic Regression, Decision Tree, Random Forest, and Neural Network classifiers.
2. SHAP-based explainability for both global model behavior and local individual predictions.
3. Fairness analysis using demographic parity, equal opportunity, and disparate impact.

The system is tested on two UCI benchmark datasets:

- Heart Disease dataset: a medical classification task for predicting the presence of heart disease.
- Adult Income dataset: a socioeconomic classification task for predicting whether income exceeds $50K.

The project also includes a Streamlit dashboard that allows users to select a dataset, run predictions, inspect SHAP explanations, compare model outputs, and evaluate fairness across sensitive groups.

The main finding is that model performance, explanation, and fairness must be evaluated together. The Random Forest model achieved the strongest predictive performance on both datasets, but the Adult Income fairness analysis revealed a significant gender disparity: the model predicted income greater than $50K for males at 26.15%, compared with only 8.99% for females. This corresponds to a disparate impact value of 0.3439, far below the commonly used fairness threshold of 0.8.

This demonstrates the central argument of the project: a model can be accurate and explainable, while still exhibiting unfair behavior. Responsible AI systems must therefore combine prediction, explanation, and fairness auditing before deployment.

---

## 2. Problem Statement

Many machine learning models operate as black boxes. They can produce a prediction, but they often do not explain why the prediction was made. This is especially problematic in high-stakes domains.

In healthcare, an unexplained heart disease prediction can reduce clinician trust. A doctor may hesitate to rely on a model if the system cannot show which clinical indicators influenced the decision.

In socioeconomic decision-making, a model trained on historical data may reproduce existing social inequalities. For example, an income prediction model can learn patterns related to gender, race, or marital structure, even when those patterns reflect historical bias rather than fair decision criteria.

The project addresses three core questions:

| Question | Why It Matters |
|---|---|
| What does the model predict? | Users need a clear outcome for an individual case. |
| Why did the model make that prediction? | Users need feature-level evidence and interpretability. |
| Is the model fair across groups? | Stakeholders must identify demographic disparities before deployment. |

The goal is to build a system that does not stop at classification accuracy. Instead, it provides a complete responsible-AI workflow.

---

## 3. Objectives

The main objectives of the project are:

1. Build a reusable machine learning pipeline for two real-world datasets.
2. Train and compare four classification models on each dataset.
3. Generate model evaluation outputs including accuracy, AUC, F1-score, confusion matrices, ROC curves, and feature importance plots.
4. Apply SHAP to explain model behavior at both global and local levels.
5. Compute fairness metrics across sensitive demographic groups.
6. Build an interactive dashboard that integrates prediction, explanation, and fairness analysis.
7. Demonstrate that high accuracy alone is insufficient for trustworthy AI.

---

## 4. Dataset Overview

### 4.1 Heart Disease Dataset

The Heart Disease dataset is based on the UCI Cleveland dataset. It contains 303 patient records and 13 clinical features. The target variable indicates whether heart disease is present.

Important features include:

- `age`: patient age
- `sex`: patient sex
- `cp`: chest pain type
- `trestbps`: resting blood pressure
- `chol`: cholesterol
- `thalach`: maximum heart rate achieved
- `exang`: exercise-induced angina
- `oldpeak`: ST depression during exercise
- `ca`: number of major vessels
- `thal`: thalassemia result

Sensitive attributes used for fairness analysis are `sex` and `age_group`.

### 4.2 Adult Income Dataset

The Adult Income dataset is a UCI Census dataset containing 32,561 records. The goal is to predict whether a person's income is greater than $50K per year.

Important features include:

- `age`
- `workclass`
- `education-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

Sensitive attributes used for fairness analysis are `sex`, `race`, and `age_group`.

The Adult Income dataset is especially important for fairness analysis because it contains visible demographic imbalance. Historical income patterns are not neutral, and models trained on this data can learn biased associations.

---

## 5. Methodology

### 5.1 Preprocessing

The project uses dataset-specific preprocessing classes built on a shared base pipeline.

For Heart Disease:

- Missing values are filled using the median.
- Duplicate rows are removed.
- Features are already numeric, so no categorical encoding is required.
- Features are scaled using StandardScaler.
- Data is split into training and testing sets using stratified sampling.

For Adult Income:

- Missing values represented as `?` are replaced and removed.
- `fnlwgt` is dropped because it is a census weighting variable rather than a predictive feature.
- Categorical variables are label-encoded.
- Additional engineered features are created, including capital gain/loss logs, age group, marital indicator, high-income occupation flag, and education-age interaction.
- Numerical features are scaled using StandardScaler.
- Data is split into training and testing sets using stratified sampling.

### 5.2 Models

Four machine learning models are trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Neural Network using MLPClassifier

The models provide a useful range of interpretability and complexity. Logistic Regression is a strong linear baseline. Decision Tree is transparent but can overfit. Random Forest gives strong performance on tabular data. Neural Network captures nonlinear interactions but is less directly interpretable.

### 5.3 Explainability with SHAP

SHAP values are used to explain model predictions. SHAP assigns each feature a contribution value that shows how much that feature pushed the prediction toward or away from the positive class.

The project generates:

- SHAP bar plots for global feature importance.
- SHAP summary plots for global distribution of feature effects.
- SHAP waterfall plots for individual predictions.
- SHAP dependence plots for feature interaction analysis.

TreeExplainer is used for tree-based models, LinearExplainer is used for Logistic Regression, and KernelExplainer is used for the Neural Network.

### 5.4 Fairness Analysis

The fairness module evaluates whether predictions differ across demographic groups.

The metrics used are:

| Metric | Meaning |
|---|---|
| Demographic Parity | Positive prediction rate for each group. |
| Equal Opportunity | True positive rate for each group. |
| Disparate Impact | Ratio between the lowest and highest positive prediction rates. |

A disparate impact value below 0.8 is commonly treated as a warning sign of potential bias.

---

## 6. Results

### 6.1 Model Performance

The Random Forest model achieved the strongest overall performance on both datasets.

#### Heart Disease Results

| Model | Accuracy | AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 86.9% | 0.951 | 0.867 |
| Decision Tree | 72.1% | 0.729 | 0.730 |
| Random Forest | 90.2% | 0.951 | 0.900 |
| Neural Network | 83.6% | 0.931 | 0.844 |

The Heart Disease task shows that Random Forest and Logistic Regression both achieved very strong AUC values. Random Forest performed best overall because it captured nonlinear relationships while maintaining strong generalization.

#### Adult Income Results

| Model | Accuracy | AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 81.8% | 0.850 | 0.547 |
| Decision Tree | 85.0% | 0.895 | 0.669 |
| Random Forest | 86.0% | 0.915 | 0.689 |
| Neural Network | 84.9% | 0.903 | 0.656 |

The Adult Income task is more imbalanced, so AUC and F1 are important alongside accuracy. Random Forest achieved the best overall performance.

---

## 7. SHAP Explainability Findings

### 7.1 Heart Disease

For the Heart Disease Random Forest model, the most influential SHAP features include:

- `thal`: thalassemia result
- `ca`: number of major vessels
- `thalach`: maximum heart rate achieved
- `oldpeak`: ST depression
- `exang`: exercise-induced angina

These features are clinically meaningful. This strengthens the trustworthiness of the model because its decisions are based on indicators that align with cardiovascular risk assessment.

The SHAP waterfall plot provides an individual-level explanation. It shows how each feature moves the prediction from the baseline risk toward the final predicted probability.

This is valuable because a clinician can inspect not only the final output, but also the reasoning behind the output.

### 7.2 Adult Income

For the Adult Income Random Forest model, important SHAP features include:

- `education-num`
- `relationship`
- `is_married`
- `capital_gain_log`
- `occupation`
- `age`

The Adult Income results show an important fairness concern. Some features, such as `relationship` and marital status indicators, may act as proxies for gender and social structure. Even if the model does not rely heavily on the direct `sex` column, it can still learn gender-linked patterns through proxy variables.

This is a key insight: low direct importance for a protected attribute does not guarantee fairness.

---

## 8. Fairness Findings

The most important fairness result appears in the Adult Income dataset.

For the Random Forest model:

| Group | Positive Prediction Rate |
|---|---:|
| Male | 26.15% |
| Female | 8.99% |

The model predicts income greater than $50K for males approximately 2.9 times more often than for females.

The disparate impact value is:

| Metric | Value |
|---|---:|
| Disparate Impact | 0.3439 |
| Fairness Threshold | 0.8 |

This is a significant fairness concern. A value of 0.3439 is far below the commonly used threshold of 0.8. Therefore, the model may be reproducing gender disparity found in the historical data.

This finding supports one of the central conclusions of the project: a model can perform well statistically while still behaving unfairly across demographic groups.

---

## 9. Dashboard Implementation

The Streamlit dashboard is organized into three main pages.

### 9.1 Prediction Page

The Prediction page allows users to:

- Select a dataset.
- Select a model.
- Load preset profiles.
- Enter or adjust feature values.
- Generate a prediction.
- View model agreement across all four models.

For Heart Disease, the dashboard includes clinical patient presets such as low-risk, borderline, high-risk, stress-test flag, and fairness-probe profiles.

For Adult Income, the dashboard includes demographic and occupational profiles that demonstrate how socioeconomic features influence the prediction.

### 9.2 Explanation Page

The Explanation page displays saved SHAP plots:

- Summary plot
- Feature importance bar plot
- Waterfall plot

This page explains both global model behavior and individual prediction behavior.

### 9.3 Fairness Page

The Fairness page computes and displays:

- Demographic parity
- Equal opportunity
- Disparate impact

It shows whether predictions differ substantially across sex, race, and age groups.

---

## 10. Discussion

The project demonstrates that machine learning evaluation must go beyond accuracy. Accuracy answers only whether the model makes correct predictions on average. It does not explain why predictions occur, and it does not reveal whether certain groups are disadvantaged.

SHAP explanations make model behavior more transparent. In the Heart Disease dataset, SHAP confirms that the model relies on medically meaningful clinical variables. In the Adult Income dataset, SHAP reveals the importance of features that can act as social proxies.

Fairness analysis adds another layer. The Adult Income model shows a large gender gap in positive predictions, even though the model performs well overall. This indicates that a model can be technically strong while still requiring fairness intervention before deployment.

Together, SHAP and fairness analysis provide a more complete picture of model behavior.

---

## 11. Limitations

The project has several limitations:

1. The Heart Disease dataset is small, with only 303 records.
2. Fairness metrics can identify disparities but do not automatically explain their social causes.
3. Label encoding may impose artificial ordering on categorical features.
4. The project audits bias but does not yet apply bias mitigation methods.
5. Neural Network SHAP explanations are slower because KernelExplainer is computationally expensive.

These limitations create clear opportunities for future improvement.

---

## 12. Future Work

Future improvements could include:

1. Applying fairness mitigation techniques such as reweighting, resampling, or threshold adjustment.
2. Using one-hot encoding or target encoding for categorical features.
3. Adding confidence intervals for fairness metrics.
4. Expanding the medical dataset to improve generalization.
5. Adding model calibration analysis.
6. Improving the dashboard with downloadable reports.
7. Adding API support for both datasets.

---

## 13. Conclusion

This project built a complete Explainable AI decision-support system for two high-impact prediction tasks. It combines model training, SHAP explainability, fairness auditing, and an interactive dashboard.

The main conclusion is clear: responsible machine learning requires more than predictive performance. A model should be accurate, explainable, and fair.

The Heart Disease results show how SHAP can make medical predictions more interpretable by highlighting clinically meaningful features. The Adult Income results show how fairness analysis can reveal demographic disparities hidden behind strong model performance.

The system demonstrates that explainability and fairness are not optional additions. They are essential components of trustworthy AI.

---

## References

1. Lundberg, S. M., and Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.
2. Becker, B., and Kohavi, R. (1996). Adult Data Set. UCI Machine Learning Repository.
3. Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease.
4. Barocas, S., Hardt, M., and Narayanan, A. Fairness and Machine Learning.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
