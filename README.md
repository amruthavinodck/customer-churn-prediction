# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, so the business can act with retention offers *before* they leave.

## Problem
Losing an existing customer costs far more than retaining one. This project builds a classification model that flags high-risk customers early, using account and billing data (contract type, tenure, monthly charges, service usage).

## Approach
1. **Data cleaning** — handled missing billing values, dropped non-predictive ID column
2. **EDA** — found churn is heavily concentrated in month-to-month, low-tenure, high-monthly-charge customers
3. **Feature encoding** — one-hot encoded categorical variables (contract, payment method, services)
4. **Modeling** — trained and compared Logistic Regression (interpretable baseline) vs Random Forest (non-linear patterns)
5. **Evaluation** — Accuracy, Precision, **Recall**, F1, ROC-AUC — prioritized Recall since missing a real churner is costlier than a false alarm
6. **Feature importance** — identified tenure, total/monthly charges, and contract length as the top churn drivers

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.70 | 0.67 | 0.60 | 0.63 | 0.76 |
| Random Forest | 0.69 | 0.63 | **0.66** | 0.64 | 0.76 |

**Random Forest chosen as the final model** — it catches more actual churners (higher recall), which matters more than raw accuracy for a retention use case.

## Key Insight
Month-to-month customers churn at **57%**, compared to **18%** for two-year contract customers. Combined with low tenure and high monthly charges, this is the highest-risk segment — a clear, actionable target for retention campaigns.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Files
- `Customer_Churn_Prediction.ipynb` — full notebook with explanations, charts, and analysis
- `churn_analysis.py` — standalone script version
- `generate_data.py` — synthetic dataset generator (realistic telecom churn patterns)
- `telecom_churn.csv` — dataset used


## What I'd add with more time
- Hyperparameter tuning (GridSearchCV)
- XGBoost/LightGBM comparison
- SHAP values for per-customer explanations
- SMOTE for class imbalance

---
*Note: dataset is synthetically generated with realistic churn logic (not scraped from a real company), built to mirror the structure of the widely-used Telco Customer Churn dataset.*
