"""
AMLGuard AI — FastAPI Backend
Real-time AML transaction risk scoring API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import json
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.features import engineer_features, get_feature_columns

# --- App Setup ---
app = FastAPI(
    title="AMLGuard AI",
    description=(
        "Nigerian Fintech AML Transaction Monitoring API. "
        "Real-time risk scoring using XGBoost + Isolation Forest ensemble."
    ),
    version="1.0.0",
    contact={
        "name": "Wisdom Okparaji",
        "url": "https://github.com/Santandave961",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../model/artifacts")
xgb_model = None
iso_model = None
feature_cols = None


@app.on_event("startup")
def load_models():
    global xgb_model, iso_model, feature_cols
    try:
        xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
        iso_model = joblib.load(f"{MODEL_DIR}/iso_forest.pkl")
        with open(f"{MODEL_DIR}/feature_cols.json") as f:
            feature_cols = json.load(f)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Model load failed: {e}")


# --- Schemas ---
class Transaction(BaseModel):
    transaction_id: str = Field(..., example="TXN-001")
    account_id: str = Field(..., example="NG1234567890")
    amount: float = Field(..., example=4850000.00)
    transaction_type: str = Field(..., example="transfer")
    sender_bank: str = Field(..., example="Kuda Bank")
    receiver_bank: str = Field(..., example="GTBank")
    location: str = Field(..., example="Lagos")
    merchant_category: str = Field(..., example="transfer")
    is_international: bool = Field(..., example=False)
    hour_of_day: int = Field(..., example=2)
    day_of_week: int = Field(..., example=1)


class RiskScore(BaseModel):
    transaction_id: str
    account_id: str
    risk_score: float
    risk_level: str
    is_flagged: bool
    aml_triggers: List[str]
    recommendation: str


class BatchRequest(BaseModel):
    transactions: List[Transaction]


# --- Risk Logic ---
def score_transaction(txn: dict) -> dict:
    df = pd.DataFrame([txn])
    df["timestamp"] = pd.Timestamp.now()

    df = engineer_features(df)
    X = df[feature_cols].fillna(0)

    # XGBoost probability
    xgb_proba = float(xgb_model.predict_proba(X)[0][1])

    # Isolation Forest score (-1 = anomaly, 1 = normal)
    iso_score = float(iso_model.decision_function(X)[0])
    iso_flag = iso_model.predict(X)[0] == -1

    # Ensemble risk score
    iso_normalized = max(0, min(1, (0.5 - iso_score)))
    risk_score = round(0.7 * xgb_proba + 0.3 * iso_normalized, 4)

    # Determine risk level
    if risk_score >= 0.75:
        risk_level = "CRITICAL"
    elif risk_score >= 0.50:
        risk_level = "HIGH"
    elif risk_score >= 0.25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # AML triggers
    triggers = []
    amount = txn["amount"]
    if 4_500_000 <= amount <= 4_999_999:
        triggers.append("structuring_threshold")
    if txn["hour_of_day"] in range(0, 5):
        triggers.append("unusual_hours")
    if txn.get("is_international"):
        triggers.append("international_transfer")
    if amount > 5_000_000:
        triggers.append("large_transaction")
    if amount % 100_000 == 0 and amount > 0:
        triggers.append("round_amount")
    if iso_flag:
        triggers.append("anomaly_detected")

    # Recommendation
    if risk_level == "CRITICAL":
        recommendation = "BLOCK — Escalate to compliance officer immediately."
    elif risk_level == "HIGH":
        recommendation = "HOLD — Requires manual review before processing."
    elif risk_level == "MEDIUM":
        recommendation = "MONITOR — Flag for enhanced due diligence."
    else:
        recommendation = "APPROVE — Transaction within normal parameters."

    return {
        "transaction_id": txn["transaction_id"],
        "account_id": txn["account_id"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "is_flagged": risk_score >= 0.50,
        "aml_triggers": triggers,
        "recommendation": recommendation,
    }


# --- Endpoints ---
@app.get("/")
def root():
    return {
        "name": "AMLGuard AI",
        "version": "1.0.0",
        "author": "github.com/Santandave961",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if xgb_model else "model_not_loaded",
        "model": "XGBoost + Isolation Forest ensemble",
        "compliance": "CBN AML Framework aligned"
    }


@app.post("/score", response_model=RiskScore)
def score(transaction: Transaction):
    if not xgb_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    result = score_transaction(transaction.dict())
    return result


@app.post("/score/batch")
def score_batch(request: BatchRequest):
    if not xgb_model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = [score_transaction(t.dict()) for t in request.transactions]
    flagged = [r for r in results if r["is_flagged"]]
    return {
        "total": len(results),
        "flagged": len(flagged),
        "flag_rate": round(len(flagged) / len(results), 4),
        "results": results
    }


@app.get("/stats")
def stats():
    """Return model performance stats."""
    try:
        with open(f"{MODEL_DIR}/metrics.json") as f:
            metrics = json.load(f)
        return metrics
    except Exception:
        return {"error": "Metrics not available"}