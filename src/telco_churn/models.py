"""Model pipelines, evaluation metrics and retention-campaign targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ]
    )


def candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Majority class (baseline)": Pipeline([("prep", preprocessor()), ("clf", DummyClassifier())]),
        "Logistic regression": Pipeline(
            [("prep", preprocessor()), ("clf", LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))]
        ),
        "Random forest": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=random_state
                    ),
                ),
            ]
        ),
        "Gradient boosting": Pipeline(
            [
                ("prep", preprocessor()),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                        class_weight="balanced", random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def classification_scores(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def lift_table(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Churn rate and cumulative share of churners captured by risk decile.

    Decile 1 holds the 10% of customers the model rates as most likely to leave.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)}).sort_values("p", ascending=False)
    df["decile"] = np.arange(len(df)) * bins // len(df) + 1
    base_rate = df["y"].mean()
    out = df.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"))
    out["churn_rate"] = out["churners"] / out["customers"]
    out["lift"] = out["churn_rate"] / base_rate
    out["cum_share_of_churners"] = out["churners"].cumsum() / out["churners"].sum()
    return out


def campaign_value(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    customer_value: float,
    offer_cost: float,
    save_rate: float,
) -> float:
    """Expected net value of offering a retention deal to everyone above ``threshold``.

    Every targeted customer costs ``offer_cost``. A targeted churner is kept with
    probability ``save_rate``, which saves ``customer_value``.
    """
    targeted = np.asarray(proba) >= threshold
    saved = np.sum(np.asarray(y_true)[targeted]) * save_rate * customer_value
    return float(saved - targeted.sum() * offer_cost)


def best_threshold(y_true, proba, customer_value, offer_cost, save_rate) -> tuple[float, float]:
    """Threshold (0.05 to 0.95) that maximises :func:`campaign_value`."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    values = [campaign_value(y_true, proba, t, customer_value, offer_cost, save_rate) for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), float(values[i])
