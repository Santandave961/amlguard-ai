"""
AMLGuard AI — Streamlit Dashboard
Real-time AML transaction monitoring dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.features import engineer_features, get_feature_columns

# --- Page Config ---
st.set_page_config(
    page_title="AMLGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    
    .metric-card {
        background: #0a0f1e;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .risk-critical { color: #ff4444; font-weight: 700; font-size: 1.4rem; }
    .risk-high { color: #ff8c00; font-weight: 700; font-size: 1.4rem; }
    .risk-medium { color: #ffd700; font-weight: 700; font-size: 1.4rem; }
    .risk-low { color: #00cc66; font-weight: 700; font-size: 1.4rem; }
    
    .stDataFrame { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model/artifacts"


@st.cache_resource
def load_models():
    xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
    iso = joblib.load(f"{MODEL_DIR}/iso_forest.pkl")
    with open(f"{MODEL_DIR}/feature_cols.json") as f:
        cols = json.load(f)
    return xgb, iso, cols


@st.cache_data
def load_data():
    df = pd.read_csv("data/transactions.csv")
    return df


def get_risk_level(score):
    if score >= 0.75: return "CRITICAL"
    if score >= 0.50: return "HIGH"
    if score >= 0.25: return "MEDIUM"
    return "LOW"


def get_risk_color(level):
    return {"CRITICAL": "#ff4444", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#00cc66"}[level]


# --- Header ---
st.markdown("# 🛡️ AMLGuard AI")
st.markdown("**Nigerian Fintech Anti-Money Laundering Monitor** | CBN Framework Aligned")
st.divider()

# --- Load ---
try:
    xgb_model, iso_model, feature_cols = load_models()
    df_raw = load_data()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}. Run `python model/train.py` first.")
    model_loaded = False
    st.stop()

# --- Score all transactions ---
@st.cache_data
def score_all(df):
    df = engineer_features(df.copy())
    X = df[feature_cols].fillna(0)
    proba = xgb_model.predict_proba(X)[:, 1]
    iso_scores = iso_model.decision_function(X)
    iso_normalized = np.clip(0.5 - iso_scores, 0, 1)
    risk_scores = 0.7 * proba + 0.3 * iso_normalized
    df["risk_score"] = risk_scores
    df["risk_level"] = df["risk_score"].apply(get_risk_level)
    return df

df = score_all(df_raw)

# --- KPI Row ---
total = len(df)
flagged = (df["risk_score"] >= 0.5).sum()
critical = (df["risk_level"] == "CRITICAL").sum()
total_flagged_amount = df[df["risk_score"] >= 0.5]["amount"].sum()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", f"{total:,}")
with col2:
    st.metric("Flagged", f"{flagged:,}", delta=f"{flagged/total*100:.1f}%", delta_color="inverse")
with col3:
    st.metric("Critical Alerts", f"{critical:,}", delta_color="inverse")
with col4:
    st.metric("Flagged Amount (NGN)", f"₦{total_flagged_amount/1e6:.1f}M")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Flagged Transactions", "🔍 Score Transaction", "📈 Model Stats"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        color_map = {"CRITICAL": "#ff4444", "HIGH": "#ff8c00", "MEDIUM": "#ffd700", "LOW": "#00cc66"}
        fig = px.bar(
            risk_counts, x="Risk Level", y="Count",
            color="Risk Level", color_discrete_map=color_map,
            title="Transaction Risk Distribution"
        )
        fig.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        flag_counts = df[df["is_suspicious"] == 1]["aml_flag"].value_counts().reset_index()
        flag_counts.columns = ["AML Pattern", "Count"]
        fig2 = px.pie(
            flag_counts, names="AML Pattern", values="Count",
            title="AML Pattern Breakdown",
            color_discrete_sequence=px.colors.sequential.Reds_r
        )
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    # Risk score distribution
    fig3 = px.histogram(
        df, x="risk_score", nbins=50,
        title="Risk Score Distribution",
        color_discrete_sequence=["#00aaff"]
    )
    fig3.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Flag Threshold")
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    flagged_df = df[df["risk_score"] >= 0.5].sort_values("risk_score", ascending=False)

    st.markdown(f"### 🚨 {len(flagged_df):,} Flagged Transactions")

    display_cols = ["transaction_id", "account_id", "amount", "transaction_type",
                    "location", "risk_score", "risk_level", "aml_flag"]

    def color_risk(val):
        colors = {"CRITICAL": "background-color: #3d0000", "HIGH": "background-color: #3d1a00",
                  "MEDIUM": "background-color: #3d3300", "LOW": "background-color: #003d1a"}
        return colors.get(val, "")

    styled = flagged_df[display_cols].head(100).style.map(
        color_risk, subset=["risk_level"]
    ).format({"risk_score": "{:.3f}", "amount": "₦{:,.0f}"})

    st.dataframe(styled, use_container_width=True)

with tab3:
    st.markdown("### 🔍 Score a New Transaction")

    col1, col2 = st.columns(2)
    with col1:
        account_id = st.text_input("Account ID", "NG1234567890")
        amount = st.number_input("Amount (NGN)", min_value=0.0, value=4850000.0, step=1000.0)
        txn_type = st.selectbox("Transaction Type", ["transfer", "airtime_purchase", "bill_payment", "pos_payment", "atm_withdrawal"])
        sender_bank = st.selectbox("Sender Bank", ["Kuda Bank", "Moniepoint", "GTBank", "Access Bank", "Zenith Bank"])

    with col2:
        receiver_bank = st.selectbox("Receiver Bank", ["GTBank", "Kuda Bank", "Access Bank", "Zenith Bank", "First Bank"])
        location = st.selectbox("Location", ["Lagos", "Abuja", "Kano", "Rivers", "Oyo"])
        hour = st.slider("Hour of Day", 0, 23, 2)
        is_international = st.checkbox("International Transfer")

    if st.button("🛡️ Score Transaction", type="primary"):
        txn = {
            "transaction_id": "TXN-LIVE-001",
            "account_id": account_id,
            "amount": amount,
            "transaction_type": txn_type,
            "sender_bank": sender_bank,
            "receiver_bank": receiver_bank,
            "location": location,
            "merchant_category": "transfer",
            "is_international": is_international,
            "hour_of_day": hour,
            "day_of_week": datetime.now().weekday(),
        }

        df_txn = pd.DataFrame([txn])
        df_txn["timestamp"] = pd.Timestamp.now()
        df_txn = engineer_features(df_txn)
        X = df_txn[feature_cols].fillna(0)

        proba = float(xgb_model.predict_proba(X)[0][1])
        iso_score = float(iso_model.decision_function(X)[0])
        iso_norm = max(0, min(1, 0.5 - iso_score))
        risk_score = round(0.7 * proba + 0.3 * iso_norm, 4)
        risk_level = get_risk_level(risk_score)
        color = get_risk_color(risk_level)

        st.markdown(f"""
        <div style='background:#0a0f1e;border:2px solid {color};border-radius:12px;padding:24px;margin-top:16px'>
            <h2 style='color:{color};margin:0'>{risk_level} RISK</h2>
            <h1 style='color:{color};margin:4px 0 16px 0;font-size:3rem'>{risk_score:.3f}</h1>
            <p style='color:#aaa'>Risk Score (0 = Clean, 1 = Suspicious)</p>
        </div>
        """, unsafe_allow_html=True)

        # Triggers
        triggers = []
        if 4_500_000 <= amount <= 4_999_999: triggers.append("⚠️ Structuring threshold")
        if hour in range(0, 5): triggers.append("🌙 Unusual hours (night transaction)")
        if is_international: triggers.append("🌍 International transfer")
        if amount > 5_000_000: triggers.append("💰 Large transaction")
        if iso_norm > 0.5: triggers.append("🔴 Anomaly detected by Isolation Forest")

        if triggers:
            st.markdown("**AML Triggers Detected:**")
            for t in triggers:
                st.warning(t)

with tab4:
    try:
        with open(f"{MODEL_DIR}/metrics.json") as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
        col2.metric("Precision", f"{metrics['precision']:.4f}")
        col3.metric("Recall", f"{metrics['recall']:.4f}")
        col4.metric("F1 Score", f"{metrics['f1_score']:.4f}")

        st.info("Model: XGBoost (70%) + Isolation Forest (30%) ensemble | Trained on Nigerian fintech transaction patterns")
    except Exception:
        st.warning("Run training first to see model stats.")

# --- Footer ---
st.divider()
st.markdown(
    "<center><small>AMLGuard AI v1.0 | Built by Wisdom Okparaji | "
    "Aligned with CBN AML/CFT Regulations</small></center>",
    unsafe_allow_html=True
)