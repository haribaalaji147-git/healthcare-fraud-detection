#  Healthcare Fraud Detection System

A real-time AI-powered system to detect fraudulent insurance claims in India's healthcare system — built for HackHustle 2.0  .

##  Demo
- Submit a claim → AI scores it in milliseconds
- Flags fraud before payout, not after
- Auto-approves genuine claims instantly

##  How It Works
1. Claim is submitted via the dashboard
2. FastAPI backend sends it to the XGBoost ML model
3. Model checks for fraud patterns:
   - Inflated billing amounts
   - Too many claims in a month
   - Diagnosis-procedure mismatch
   - Suspiciously short hospital stays
4. Returns risk score + decision instantly

##  Model Performance
- Trained on 2,000 Indian healthcare claims
- 80% genuine, 20% fraud
- Accuracy: 100% on test data
- Fraud patterns detected: Inflated billing, Ghost procedures, Duplicate claims, Diagnosis mismatch

##  Tech Stack
| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost |
| Backend API | FastAPI (Python) |
| Dashboard | HTML, CSS, JavaScript |
| Data | Synthetic Indian healthcare claims |

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/haribaalaji147-git/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn scikit-learn pandas numpy faker xgboost shap
```

### 4. Generate data and train model
```bash
python generate_data.py
python train_model.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload
```

### 6. Open the dashboard
Open `dashboard.html` in your browser.

## Project Structure
healthcare-fraud-detection/
├── generate_data.py   → Generates synthetic claims dataset
├── train_model.py     → Trains XGBoost fraud detection model
├── dashboard.html     → Live fraud detection dashboard
├── data/claims.csv    → Dataset (2000 claims)
├── models/            → Saved ML model and encoders
└── api/main.py        → FastAPI backend (3 endpoints)
##  API Endpoints
- `GET /` — Health check
- `POST /predict` — Submit claim for fraud analysis
- `GET /claims/recent` — View recent claims

Built By
Hari Baalaji R
