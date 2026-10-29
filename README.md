# Mortgage Default Prediction

A credit risk modelling project that predicts the probability that a mortgage applicant defaults. The notebook covers the full workflow: EDA, preprocessing, model training, tuning, threshold selection, calibration and explainability.

## Overview

- **Task:** Binary classification (`target_var`: 0 = non-default, 1 = default)
- **Data:** About 10,000 mortgage applications, with a default rate of about 15.4%
- **Models:** Logistic Regression (interpretable baseline) and XGBoost (flexible model)
- **Goal:** Support approval and manual-review decisions with a model that is explainable, leakage-aware and well calibrated

## Approach

1. **EDA:** Missing values, class imbalance, numerical and categorical patterns, observation dates and correlations.
2. **Preprocessing:** Rows with a missing target or date are dropped. A time-based split is used (70% train, 15% validation, 15% test). Numerical features use median imputation. Categorical features use an "Unknown" category and one-hot encoding. For Logistic Regression, features are also winsorised at the 1st and 99th percentiles and scaled. All transformers are fitted on the training set only to avoid leakage.
3. **Imbalance handling:** `class_weight="balanced"` for Logistic Regression and `scale_pos_weight` for XGBoost.
4. **Tuning:** 5-fold stratified cross-validation with F1 as the scoring metric.
5. **Threshold selection:** Maximise F1 on the validation set while keeping recall at or above 0.80.
6. **Calibration:** Platt Scaling for Logistic Regression and Isotonic Regression for XGBoost.
7. **Explainability:** Logistic Regression coefficients and XGBoost SHAP values.

## Results (test set, calibrated)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression + Platt | 0.25 | 0.503 | 0.793 | 0.616 | 0.920 | 0.688 | 0.073 |
| XGBoost + Isotonic | 0.34 | 0.443 | 0.829 | 0.578 | 0.912 | 0.634 | 0.078 |

**Recommendation:** Logistic Regression with Platt Scaling. It has the better F1, precision, ROC-AUC, PR-AUC and Brier score, and is easier to explain. XGBoost catches more defaulters but flags more good applicants. In practice, flagged applicants should go to manual review rather than being rejected automatically.

## Repository Contents

- `mortgage_default_professional.ipynb`: the full analysis notebook
- `README.md`: this file

The dataset (`Quant_Challenge_data_amended.csv`) is not included. Place it in the same folder as the notebook before running.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter notebook mortgage_default_professional.ipynb
```

Run all cells from top to bottom.

## Notes

- Input variables are assumed to be available at or before the observation date. This should be confirmed with the data owner before production use.
- The `occup` column contains unlabelled numeric codes (1, 2, 3), which are kept as separate categories because no data dictionary was provided.
- Variables such as marital status may raise fairness or regulatory considerations in a real credit decisioning setting.
