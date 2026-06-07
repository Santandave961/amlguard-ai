# 🛡️ AMLGuard AI

**Nigerian Fintech Anti-Money Laundering Transaction Monitoring System**

Real-time AML detection API + dashboard using XGBoost and Isolation Forest ensemble, aligned with CBN AML/CFT regulations.

---

## 🎯 What It Does

- Scores transactions in real-time for AML risk (0–1)
- Detects 5 AML patterns: structuring, rapid movement, unusual hours, round-tripping, velocity abuse
- Ensemble model: XGBoost (supervised) + Isolation Forest (unsupervised anomaly)
- REST API with FastAPI + interactive Streamlit dashboard
- MLflow experiment tracking

---

## 🏗️ Project Structure

```
amlguard-ai/
├── app/
│   ├── main.py              # FastAPI backend
│   └── streamlit_app.py     # Streamlit dashboard
├── data/
│   └── generate_transactions.py  # Nigerian transaction simulator
├── model/
│   ├── features.py          # Feature engineering
│   ├── train.py             # Model training + MLflow
│   └── artifacts/           # Saved models (after training)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
cd data && python generate_transactions.py && cd ..

# 3. Train models
cd model && python train.py && cd ..

# 4. Run API
uvicorn app.main:app --reload

# 5. Run Dashboard
streamlit run app/streamlit_app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/score` | Score single transaction |
| POST | `/score/batch` | Score batch of transactions |
| GET | `/stats` | Model performance metrics |

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| AUC-ROC | ~0.96 |
| Precision | ~0.89 |
| Recall | ~0.87 |
| F1 Score | ~0.88 |

---

## 🇳🇬 AML Patterns Detected

1. **Structuring** — Transactions just below NGN 5M reporting threshold
2. **Rapid Movement** — Large amounts moved in short windows
3. **Unusual Hours** — Large transactions between 12AM–5AM
4. **Round Tripping** — Funds sent and returned in circular pattern
5. **Velocity Abuse** — High frequency small transactions

---

## 👤 Author

**Wisdom Okparaji** | Data Scientist & ML Engineer  
GitHub: [Santandave961](https://github.com/Santandave961)  
LinkedIn: [wisdom-okparaji](https://linkedin.com/in/wisdom-okparaji-680550246)
