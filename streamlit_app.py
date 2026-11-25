"""Streamlit app: churn scoring and retention campaign planner.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from telco_churn.data import ADD_ONS, CATEGORICAL, NUMERIC, TARGET, clean, load  # noqa: E402
from telco_churn.models import best_threshold, campaign_value, candidate_models, lift_table  # noqa: E402

st.set_page_config(page_title="Telco Churn", page_icon="📞", layout="wide")


@st.cache_resource
def train():
    """Logistic regression: it tied with the random forest in cross-validation
    (ROC-AUC 0.8472 against 0.8474) and loads much faster."""
    df = load()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = candidate_models()["Logistic regression"].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    return model, df, X_te, y_te.to_numpy(), proba


model, df, X_te, y_te, proba = train()

st.title("📞 Telco Customer Churn")
st.caption("7,043 customers of a telecom provider, 26.5% of whom cancelled. Model: logistic regression, tested on 1,409 held-out customers.")

tab1, tab2, tab3 = st.tabs(["Score a customer", "Plan a retention campaign", "What drives churn"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        monthly = st.slider("Monthly charges (USD)", 18.0, 120.0, 75.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", sorted(df["payment_method"].unique()))
    with c2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
    with c3:
        st.write("Services")
        phone = st.checkbox("Phone service", value=True)
        extras = {a: st.checkbox(a.replace("_", " ").capitalize(), value=False, key=a) for a in ADD_ONS if a not in ("phone_service", "multiple_lines")}
        multiple = st.checkbox("Multiple lines", value=False) if phone else False

    def yn(flag: bool, needs_internet: bool = True) -> str:
        if needs_internet and internet == "No":
            return "No internet service"
        return "Yes" if flag else "No"

    raw = pd.DataFrame([{
        "customerID": "new", "gender": "Female", "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes" if phone else "No",
        "MultipleLines": ("Yes" if multiple else "No") if phone else "No phone service",
        "InternetService": internet,
        "OnlineSecurity": yn(extras["online_security"]), "OnlineBackup": yn(extras["online_backup"]),
        "DeviceProtection": yn(extras["device_protection"]), "TechSupport": yn(extras["tech_support"]),
        "StreamingTV": yn(extras["streaming_tv"]), "StreamingMovies": yn(extras["streaming_movies"]),
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": str(round(monthly * tenure, 2)) if tenure else " ", "Churn": "No",
    }])
    row = clean(raw)
    score = float(model.predict_proba(row[NUMERIC + CATEGORICAL])[0, 1])
    pct_rank = float((proba < score).mean())
    m1, m2 = st.columns(2)
    m1.metric("Churn risk score", f"{score:.2f}")
    m2.metric("Riskier than", f"{pct_rank:.0%} of test customers")
    st.progress(min(max(score, 0.0), 1.0))

with tab2:
    st.write("Set your own assumptions. The value updates for the 1,409 test customers.")
    a1, a2, a3 = st.columns(3)
    months = a1.slider("Months of bill a saved customer is worth", 1, 36, 12)
    cost = a2.slider("Cost of one retention offer (USD)", 5, 200, 50)
    save = a3.slider("Share of targeted churners who stay", 0.05, 0.8, 0.30, step=0.05)
    value = float((X_te["monthly_charges"] * months).mean())
    best_t, best_v = best_threshold(y_te, proba, value, cost, save)
    thr = st.slider("Send an offer when the score is at least", 0.05, 0.95, float(best_t), step=0.01)
    v = campaign_value(y_te, proba, thr, value, cost, save)
    everyone = campaign_value(y_te, proba, 0.0, value, cost, save)
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers targeted", f"{int((proba >= thr).sum()):,}")
    k2.metric("Net campaign value", f"${v:,.0f}", delta=f"${v - everyone:,.0f} vs offering everyone")
    k3.metric("Best threshold for these assumptions", f"{best_t:.2f}", help=f"Net value ${best_v:,.0f}")
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = pd.DataFrame({"threshold": grid, "Net value (USD thousands)": [campaign_value(y_te, proba, t, value, cost, save) / 1000 for t in grid]})
    st.line_chart(curve, x="threshold", y="Net value (USD thousands)")

with tab3:
    d1, d2 = st.columns(2)
    with d1:
        st.write("Churn rate by contract")
        st.bar_chart(df.groupby("contract")[TARGET].mean().mul(100).rename("Churn rate (%)"))
    with d2:
        st.write("Churn rate by tenure band")
        order = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
        st.bar_chart(df.groupby("tenure_band")[TARGET].mean().reindex(order).mul(100).rename("Churn rate (%)"))
    lt = lift_table(y_te, proba)
    st.write("Actual churn rate by predicted risk decile (1 = highest risk)")
    st.bar_chart(lt["churn_rate"].mul(100).rename("Churn rate (%)"))

st.markdown("[Source code and README](https://github.com/Collinsthe4th/telco-customer-churn)")
