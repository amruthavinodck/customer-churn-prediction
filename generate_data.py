"""
generate_data.py
-----------------
Creates a realistic synthetic telecom customer dataset with churn patterns
(similar structure to the popular Telco Customer Churn dataset on Kaggle).

Why synthetic? So we control the ground-truth patterns and can EXPLAIN
exactly why the model learns what it learns -- great for interviews.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

customer_id = [f"CUST{1000+i}" for i in range(n)]
gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])

tenure = np.random.randint(0, 73, n)  # months with company

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21]
)
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21]
)
tech_support = np.random.choice(["Yes", "No"], n, p=[0.29, 0.71])
online_security = np.random.choice(["Yes", "No"], n, p=[0.29, 0.71])

monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
total_charges = np.clip(total_charges, 0, None)

# ---- Build churn probability from realistic business logic ----
# Month-to-month + fiber + high charges + low tenure + no tech support => higher churn risk
churn_prob = 0.05
churn_prob += (contract == "Month-to-month") * 0.30
churn_prob += (contract == "One year") * 0.08
churn_prob += (internet_service == "Fiber optic") * 0.15
churn_prob += (tech_support == "No") * 0.10
churn_prob += (online_security == "No") * 0.08
churn_prob += (payment_method == "Electronic check") * 0.10
churn_prob += (tenure < 12) * 0.15
churn_prob += (monthly_charges > 80) * 0.10
churn_prob -= (tenure > 48) * 0.20
churn_prob -= (contract == "Two year") * 0.15
churn_prob = np.clip(churn_prob, 0.02, 0.95)

churn = np.random.binomial(1, churn_prob)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "OnlineSecurity": online_security,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_label,
})

# introduce a few realistic missing values (common in real datasets)
missing_idx = np.random.choice(df.index, 15, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

df.to_csv("/home/claude/churn_project/telecom_churn.csv", index=False)
print("Dataset created:", df.shape)
print(df["Churn"].value_counts(normalize=True))
