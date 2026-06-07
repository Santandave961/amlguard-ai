"""
AMLGuard AI — Nigerian Transaction Data Generator
Simulates realistic Nigerian fintech transaction patterns
including both normal and suspicious/AML scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

random.seed(42)
np.random.seed(42)

# Nigerian-specific constants
NIGERIAN_BANKS = [
    "Kuda Bank", "Moniepoint", "OPay", "PalmPay", "Carbon",
    "GTBank", "Access Bank", "Zenith Bank", "First Bank", "UBA",
    "Sterling Bank", "Wema Bank", "Stanbic IBTC", "Flutterwave"
]

TRANSACTION_TYPES = [
    "transfer", "airtime_purchase", "bill_payment", "pos_payment",
    "atm_withdrawal", "ussd_transfer", "mobile_top_up", "merchant_payment"
]

NIGERIAN_STATES = [
    "Lagos", "Abuja", "Kano", "Rivers", "Oyo", "Delta",
    "Anambra", "Enugu", "Kaduna", "Ogun"
]

MERCHANT_CATEGORIES = [
    "supermarket", "restaurant", "fuel_station", "pharmacy",
    "electronics", "clothing", "transport", "utility"
]


def generate_account_id():
    return f"NG{random.randint(1000000000, 9999999999)}"


def generate_normal_transaction(account_id, timestamp):
    """Generate a normal, legitimate transaction."""
    amount = np.random.lognormal(mean=9.5, sigma=1.2)  # NGN amounts
    amount = round(min(amount, 500_000), 2)  # Cap at 500k NGN

    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account_id,
        "timestamp": timestamp,
        "amount": amount,
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "sender_bank": random.choice(NIGERIAN_BANKS),
        "receiver_bank": random.choice(NIGERIAN_BANKS),
        "location": random.choice(NIGERIAN_STATES),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "is_international": random.random() < 0.05,
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "is_suspicious": 0,
        "aml_flag": "clean"
    }


def generate_suspicious_transaction(account_id, timestamp, pattern_type):
    """Generate suspicious transaction with specific AML pattern."""

    if pattern_type == "structuring":
        # Transactions just below reporting threshold (NGN 5M)
        amount = random.uniform(4_700_000, 4_990_000)
        flag = "structuring"

    elif pattern_type == "rapid_movement":
        # Large amount moved quickly
        amount = random.uniform(1_000_000, 10_000_000)
        flag = "rapid_movement"

    elif pattern_type == "unusual_hours":
        # Late night large transaction
        timestamp = timestamp.replace(hour=random.randint(1, 4))
        amount = random.uniform(500_000, 5_000_000)
        flag = "unusual_hours"

    elif pattern_type == "round_tripping":
        # Same amount sent and received
        amount = round(random.uniform(100_000, 2_000_000) / 1000) * 1000
        flag = "round_tripping"

    elif pattern_type == "velocity":
        # Many small transactions in short time
        amount = random.uniform(1_000, 50_000)
        flag = "velocity"

    else:
        amount = random.uniform(500_000, 5_000_000)
        flag = "unusual_pattern"

    return {
        "transaction_id": str(uuid.uuid4()),
        "account_id": account_id,
        "timestamp": timestamp,
        "amount": round(amount, 2),
        "transaction_type": random.choice(["transfer", "ussd_transfer"]),
        "sender_bank": random.choice(NIGERIAN_BANKS),
        "receiver_bank": random.choice(NIGERIAN_BANKS),
        "location": random.choice(NIGERIAN_STATES),
        "merchant_category": "transfer",
        "is_international": random.random() < 0.3,
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "is_suspicious": 1,
        "aml_flag": flag
    }


def generate_dataset(n_accounts=500, days=90, suspicious_ratio=0.08):
    """Generate full transaction dataset."""
    print(f"Generating transactions for {n_accounts} accounts over {days} days...")

    accounts = [generate_account_id() for _ in range(n_accounts)]
    start_date = datetime.now() - timedelta(days=days)
    transactions = []

    suspicious_patterns = [
        "structuring", "rapid_movement", "unusual_hours",
        "round_tripping", "velocity"
    ]

    # Mark some accounts as high-risk
    suspicious_accounts = set(random.sample(accounts, int(n_accounts * 0.15)))

    for account in accounts:
        # Each account has 10-100 transactions
        n_txns = random.randint(10, 100)
        is_suspicious_account = account in suspicious_accounts

        for _ in range(n_txns):
            days_offset = random.randint(0, days)
            hours_offset = random.randint(0, 23)
            timestamp = start_date + timedelta(days=days_offset, hours=hours_offset)

            # Suspicious accounts have higher chance of flagged transactions
            if is_suspicious_account and random.random() < 0.35:
                pattern = random.choice(suspicious_patterns)
                txn = generate_suspicious_transaction(account, timestamp, pattern)
            elif random.random() < suspicious_ratio:
                pattern = random.choice(suspicious_patterns)
                txn = generate_suspicious_transaction(account, timestamp, pattern)
            else:
                txn = generate_normal_transaction(account, timestamp)

            transactions.append(txn)

    df = pd.DataFrame(transactions)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Generated {len(df):,} transactions")
    print(f"Suspicious: {df['is_suspicious'].sum():,} ({df['is_suspicious'].mean()*100:.1f}%)")
    print(f"\nAML Flag Distribution:")
    print(df['aml_flag'].value_counts())

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("transactions.csv", index=False)
    print("\nSaved to transactions.csv")