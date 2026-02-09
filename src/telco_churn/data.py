"""Loading, cleaning and feature preparation for the Telco churn data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_churn.csv"

TARGET = "churn"

NUMERIC = ["tenure", "monthly_charges", "total_charges", "avg_monthly_spend", "n_services"]

CATEGORICAL = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_band",
]

ADD_ONS = [
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

RENAME = {
    "customerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame with engineered features and a 0/1 target.

    ``TotalCharges`` is stored as text and is blank for 11 brand-new customers
    (tenure 0). They have not been billed yet, so their total is set to 0.
    """
    df = raw.rename(columns=RENAME).copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0.0)
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET].str.strip() == "Yes").astype(int)

    # Engineered features.
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["total_charges"] / df["tenure"].clip(lower=1), df["monthly_charges"])
    df["n_services"] = (df[ADD_ONS] == "Yes").sum(axis=1)
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
    ).astype(str)
    return df


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See the README for the data source.")
    return clean(pd.read_csv(path))
