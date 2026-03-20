import numpy as np
import pandas as pd
import pytest

from telco_churn.data import CATEGORICAL, NUMERIC, clean
from telco_churn.models import best_threshold, campaign_value, candidate_models, lift_table


def raw_rows() -> pd.DataFrame:
    base = {
        "customerID": "x", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": "500", "Churn": "No",
    }
    new = {**base, "customerID": "y", "tenure": 0, "TotalCharges": " ", "Churn": "Yes"}
    return pd.DataFrame([base, new])


def test_clean_handles_blank_total_charges_and_features():
    df = clean(raw_rows())
    assert df["total_charges"].tolist() == [500.0, 0.0]
    assert df["churn"].tolist() == [0, 1]
    assert df.loc[0, "n_services"] == 3  # phone, online security and tech support
    assert df.loc[1, "avg_monthly_spend"] == 50.0
    assert set(NUMERIC + CATEGORICAL) <= set(df.columns)


def test_lift_table_orders_by_risk():
    y = np.array([1] * 10 + [0] * 90)
    p = np.linspace(1, 0, 100)
    lt = lift_table(y, p)
    assert lt.loc[1, "churn_rate"] == 1.0
    assert lt.loc[1, "lift"] == pytest.approx(10.0)
    assert lt["cum_share_of_churners"].iloc[-1] == pytest.approx(1.0)


def test_campaign_value_and_threshold():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    # Targeting the top two saves 2 * 0.5 * 100 = 100 and costs 2 * 10.
    assert campaign_value(y, p, 0.5, 100, 10, 0.5) == pytest.approx(80)
    thr, val = best_threshold(y, p, 100, 10, 0.5)
    assert 0.3 < thr <= 0.8 and val == pytest.approx(80)


def test_pipelines_fit_on_clean_data():
    rng = np.random.default_rng(0)
    rows = pd.concat([raw_rows()] * 40, ignore_index=True)
    rows["tenure"] = rng.integers(1, 72, len(rows))
    rows["TotalCharges"] = (rows["tenure"] * 50).astype(str)
    rows["Churn"] = np.where(rows["tenure"] < 20, "Yes", "No")
    df = clean(rows)
    model = candidate_models()["Logistic regression"].fit(df[NUMERIC + CATEGORICAL], df["churn"])
    proba = model.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    assert proba[df["tenure"] < 10].mean() > proba[df["tenure"] > 40].mean()
