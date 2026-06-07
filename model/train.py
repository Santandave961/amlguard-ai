"""
AMLGuard AI — Model Training
XGBoost + Isolation Forest ensemble for AML detection.
Tracked with MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import engineer_features, get_feature_columns

# Paths
DATA_PATH = "../data/transactions.csv"
MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df["is_suspicious"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Positive rate: {y.mean()*100:.1f}%")

    mlflow.set_experiment("amlguard-ai")

    with mlflow.start_run(run_name="xgboost_aml_detector"):

        # --- XGBoost Classifier ---
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            use_label_encoder=False
        )
        xgb.fit(X_train, y_train)

        # --- Isolation Forest (unsupervised anomaly layer) ---
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42
        )
        iso.fit(X_train)

        # --- Evaluate XGBoost ---
        y_pred = xgb.predict(X_test)
        y_proba = xgb.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\nXGBoost Results:")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Suspicious"]))

        # --- Log to MLflow ---
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": round(scale_pos_weight, 2),
        })
        mlflow.log_metrics({
            "auc_roc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        # --- Save models ---
        joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")
        joblib.dump(iso, f"{MODEL_DIR}/iso_forest.pkl")

        # Save feature columns
        with open(f"{MODEL_DIR}/feature_cols.json", "w") as f:
            json.dump(feature_cols, f)

        # Save metrics
        metrics = {
            "auc_roc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        with open(f"{MODEL_DIR}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.sklearn.log_model(iso, "isolation_forest")
        mlflow.xgboost.log_model(xgb, "xgboost_model")

        print(f"\nModels saved to {MODEL_DIR}/")
        print("MLflow run logged.")

    return xgb, iso


if __name__ == "__main__":
    train()