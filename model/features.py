"""
AMLGuard AI — Feature Engineering
Builds transaction features for AML detection.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from raw transaction data."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["account_id", "timestamp"])

    # --- Time features ---
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)

    # --- Amount features ---
    df["amount_log"] = np.log1p(df["amount"])
    df["near_threshold"] = (df["amount"].between(4_500_000, 4_999_999)).astype(int)
    df["is_round_amount"] = (df["amount"] % 100_000 == 0).astype(int)

    # --- Account-level rolling features ---
    df = df.sort_values(["account_id", "timestamp"])

    # Rolling 24h transaction count per account
    df["txn_count_24h"] = (
        df.groupby("account_id")["transaction_id"]
        .transform(lambda x: x.expanding().count())
    )

    # Rolling 7-day amount sum per account
    df["amount_sum_7d"] = (
        df.groupby("account_id")["amount"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )

    # Average transaction amount per account
    df["avg_txn_amount"] = (
        df.groupby("account_id")["amount"]
        .transform(lambda x: x.expanding().mean())
    )

    # Amount deviation from account average
    df["amount_zscore"] = (
        (df["amount"] - df["avg_txn_amount"]) /
        (df.groupby("account_id")["amount"].transform("std").fillna(1))
    )

    # --- Velocity features ---
    df["velocity_flag"] = (df["txn_count_24h"] > 20).astype(int)

    # --- International flag ---
    df["is_international"] = df["is_international"].astype(int)

    # --- Encode categoricals ---
    df["transaction_type_enc"] = pd.Categorical(df["transaction_type"]).codes
    df["sender_bank_enc"] = pd.Categorical(df["sender_bank"]).codes

    return df


def get_feature_columns():
    return [
        "amount_log",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "near_threshold",
        "is_round_amount",
        "txn_count_24h",
        "amount_sum_7d",
        "amount_zscore",
        "velocity_flag",
        "is_international",
        "transaction_type_enc",
        "sender_bank_enc",
    ]