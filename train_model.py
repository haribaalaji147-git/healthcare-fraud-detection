import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os

print("Loading data...")
df = pd.read_csv("data/claims.csv")

# Convert text columns to numbers (ML only understands numbers)
le_hospital = LabelEncoder()
le_procedure = LabelEncoder()
le_diagnosis = LabelEncoder()

df["hospital_enc"] = le_hospital.fit_transform(df["hospital"])
df["procedure_enc"] = le_procedure.fit_transform(df["procedure"])
df["diagnosis_enc"] = le_diagnosis.fit_transform(df["diagnosis"])

# Features the model will learn from
features = [
    "patient_age",
    "claim_amount",
    "days_in_hospital",
    "claims_in_month",
    "hospital_enc",
    "procedure_enc",
    "diagnosis_enc",
]

X = df[features]
y = df["is_fraud"]

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training fraud detection model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Test how good it is
y_pred = model.predict(X_test)
print("\n--- Model Performance ---")
print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))

# Save the model and encoders for the API to use
os.makedirs("models", exist_ok=True)
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/encoders.pkl", "wb") as f:
    pickle.dump({
        "hospital": le_hospital,
        "procedure": le_procedure,
        "diagnosis": le_diagnosis,
    }, f)

print("\nModel saved to models/fraud_model.pkl")
print("Encoders saved to models/encoders.pkl")
print("Done!")