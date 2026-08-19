"""
Customer Churn Prediction — End-to-End Analysis
=================================================
Amrutha Vinod

Goal: Predict which customers are likely to churn (cancel service) so the
business can target them with retention offers BEFORE they leave.

Pipeline: Load -> Clean -> EDA -> Encode -> Train/Test Split ->
          Logistic Regression + Random Forest -> Evaluate -> Feature Importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

sns.set_style("whitegrid")
OUT = "/home/claude/churn_project"

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv(f"{OUT}/telecom_churn.csv")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

# TotalCharges has some missing values -> fill with median (robust to outliers)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop ID column (not predictive)
df_model = df.drop(columns=["customerID"])

print("\nMissing values after cleaning:", df_model.isnull().sum().sum())

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Overall churn rate
df["Churn"].value_counts().plot(kind="bar", ax=axes[0, 0], color=["#2E86AB", "#E63946"])
axes[0, 0].set_title("Overall Churn Distribution")
axes[0, 0].set_xlabel("Churn")

# Churn by contract type
sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 1], palette=["#2E86AB", "#E63946"])
axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].tick_params(axis="x", rotation=15)

# Tenure distribution by churn
sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[1, 0], palette=["#2E86AB", "#E63946"])
axes[1, 0].set_title("Tenure Distribution by Churn")

# Monthly charges by churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1, 1], palette=["#2E86AB", "#E63946"])
axes[1, 1].set_title("Monthly Charges by Churn")

plt.tight_layout()
plt.savefig(f"{OUT}/eda_overview.png", dpi=120)
plt.close()
print("\nSaved eda_overview.png")

# Key EDA insight (print for the story)
churn_by_contract = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean())
print("\nChurn rate by contract type:\n", churn_by_contract.round(3))

# ---------------------------------------------------------------
# 4. ENCODING CATEGORICAL VARIABLES
# ---------------------------------------------------------------
target = "Churn"
y = (df_model[target] == "Yes").astype(int)
X = df_model.drop(columns=[target])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\nCategorical columns encoded: {categorical_cols}")
print(f"Final feature count: {X_encoded.shape[1]}")

# ---------------------------------------------------------------
# 5. TRAIN/TEST SPLIT + SCALING
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 6. MODEL 1 — LOGISTIC REGRESSION (baseline, interpretable)
# ---------------------------------------------------------------
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_preds = log_model.predict(X_test_scaled)
log_proba = log_model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------
# 7. MODEL 2 — RANDOM FOREST (usually stronger, non-linear)
# ---------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    print(f"\n--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall   :", round(recall_score(y_true, y_pred), 3))
    print("F1 Score :", round(f1_score(y_true, y_pred), 3))
    print("ROC-AUC  :", round(roc_auc_score(y_true, y_proba), 3))
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

results = []
results.append(evaluate("Logistic Regression", y_test, log_preds, log_proba))
results.append(evaluate("Random Forest", y_test, rf_preds, rf_proba))

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/model_results.csv", index=False)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, preds) in zip(axes, [("Logistic Regression", log_preds), ("Random Forest", rf_preds)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    ax.set_title(f"{name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrices.png", dpi=120)
plt.close()
print("\nSaved confusion_matrices.png")

# ROC curves
plt.figure(figsize=(6, 5))
for name, proba in [("Logistic Regression", log_proba), ("Random Forest", rf_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curve.png", dpi=120)
plt.close()
print("Saved roc_curve.png")

# ---------------------------------------------------------------
# 9. FEATURE IMPORTANCE (Random Forest) — "why" the model predicts churn
# ---------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind="barh", color="#E63946")
plt.title("Top 10 Features Driving Churn Prediction (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=120)
plt.close()
print("\nSaved feature_importance.png")

print("\nTop 10 features driving churn:\n", top_features.round(3))

print("\n\n=== DONE: All outputs saved in", OUT, "===")
