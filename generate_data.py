import pandas as pd
import numpy as np
from faker import Faker
import random
import os

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

# Indian hospitals and providers
hospitals = [
    "Apollo Hospital", "Fortis Healthcare", "AIIMS Delhi",
    "Manipal Hospital", "Narayana Health", "Max Hospital",
    "Medanta", "Kokilaben Hospital", "Christian Medical College",
    "NIMHANS", "Lilavati Hospital", "Aster CMI"
]

procedures = {
    "Appendectomy": (25000, 60000),
    "Knee Replacement": (150000, 350000),
    "Angioplasty": (200000, 500000),
    "Cataract Surgery": (20000, 50000),
    "Dialysis": (2000, 5000),
    "MRI Scan": (5000, 15000),
    "CT Scan": (3000, 8000),
    "Normal Delivery": (15000, 40000),
    "C-Section": (40000, 90000),
    "Chemotherapy": (50000, 200000),
}

diagnoses = {
    "Appendectomy": "Appendicitis",
    "Knee Replacement": "Osteoarthritis",
    "Angioplasty": "Coronary Artery Disease",
    "Cataract Surgery": "Cataract",
    "Dialysis": "Chronic Kidney Disease",
    "MRI Scan": "Back Pain",
    "CT Scan": "Head Injury",
    "Normal Delivery": "Pregnancy",
    "C-Section": "Pregnancy Complications",
    "Chemotherapy": "Cancer",
}

def generate_claim(is_fraud=False):
    procedure = random.choice(list(procedures.keys()))
    min_amt, max_amt = procedures[procedure]
    hospital = random.choice(hospitals)

    if is_fraud:
        fraud_type = random.choice([
            "inflated", "ghost", "duplicate", "mismatch"
        ])

        if fraud_type == "inflated":
            # Bill 3-5x the normal amount
            amount = random.uniform(max_amt * 3, max_amt * 5)
            diagnosis = diagnoses[procedure]

        elif fraud_type == "ghost":
            # Procedure never happened - random high amount
            amount = random.uniform(min_amt, max_amt)
            diagnosis = diagnoses[procedure]

        elif fraud_type == "duplicate":
            # Same procedure billed multiple times
            amount = random.uniform(min_amt, max_amt)
            diagnosis = diagnoses[procedure]

        elif fraud_type == "mismatch":
            # Diagnosis doesn't match procedure
            wrong_procedure = random.choice(list(procedures.keys()))
            while wrong_procedure == procedure:
                wrong_procedure = random.choice(list(procedures.keys()))
            diagnosis = diagnoses[wrong_procedure]  # Wrong diagnosis!
            amount = random.uniform(min_amt, max_amt)

        claims_in_month = random.randint(8, 20)  # Suspiciously many
        days_in_hospital = random.randint(0, 1)   # Too short for procedure

    else:
        amount = random.uniform(min_amt, max_amt)
        diagnosis = diagnoses[procedure]
        claims_in_month = random.randint(1, 4)
        days_in_hospital = random.randint(2, 10)

    return {
        "claim_id": fake.uuid4(),
        "patient_name": fake.name(),
        "patient_age": random.randint(18, 85),
        "hospital": hospital,
        "procedure": procedure,
        "diagnosis": diagnosis,
        "claim_amount": round(amount, 2),
        "days_in_hospital": days_in_hospital,
        "claims_in_month": claims_in_month,
        "is_fraud": int(is_fraud)
    }

# Generate 2000 claims: 80% genuine, 20% fraud
claims = []
for _ in range(1600):
    claims.append(generate_claim(is_fraud=False))
for _ in range(400):
    claims.append(generate_claim(is_fraud=True))

random.shuffle(claims)
df = pd.DataFrame(claims)

os.makedirs("data", exist_ok=True)
df.to_csv("data/claims.csv", index=False)

print(f"Dataset created: {len(df)} claims")
print(f"Fraud claims: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.1f}%)")
print(f"Saved to data/claims.csv")