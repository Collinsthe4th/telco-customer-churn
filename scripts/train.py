"""Select a churn model with cross-validation, test it once, and size a retention campaign.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from telco_churn.data import CATEGORICAL, NUMERIC, TARGET, load  # noqa: E402
from telco_churn.models import (  # noqa: E402
    best_threshold,
    campaign_value,
    candidate_models,
    classification_scores,
    lift_table,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# Campaign assumptions (stated in the README, change them here).
MONTHS_OF_VALUE = 12   # a saved customer is worth 12 months of their bill
OFFER_COST = 50.0      # cost of one retention offer, in USD
SAVE_RATE = 0.30       # share of targeted churners who stay because of the offer


def main() -> None:
    df = load()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = {}
    for name, model in candidate_models().items():
        s = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_auc[name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}
        print(f"{name:28s} CV ROC-AUC {s.mean():.4f} ± {s.std():.4f}")

    best = max((n for n in cv_auc if "baseline" not in n), key=lambda n: cv_auc[n]["mean"])
    model = candidate_models()[best].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    test_scores = {k: round(v, 4) for k, v in classification_scores(y_test, proba).items()}

    lifts = lift_table(y_test.to_numpy(), proba)
    value = (X_test["monthly_charges"] * MONTHS_OF_VALUE).mean()
    thr, net = best_threshold(y_test.to_numpy(), proba, value, OFFER_COST, SAVE_RATE)
    target_all = campaign_value(y_test.to_numpy(), proba, 0.0, value, OFFER_COST, SAVE_RATE)

    metrics = {
        "rows": int(len(df)),
        "churn_rate": round(float(y.mean()), 4),
        "test_rows": int(len(y_test)),
        "cv_roc_auc": cv_auc,
        "best_model": best,
        "test_scores_at_0.5": test_scores,
        "top_decile_lift": round(float(lifts.loc[1, "lift"]), 2),
        "top_3_deciles_share_of_churners": round(float(lifts.loc[3, "cum_share_of_churners"]), 4),
        "campaign": {
            "assumptions": {
                "customer_value_usd": round(float(value), 2),
                "offer_cost_usd": OFFER_COST,
                "save_rate": SAVE_RATE,
            },
            "best_threshold": thr,
            "customers_targeted": int((proba >= thr).sum()),
            "net_value_usd": round(net, 2),
            "net_value_if_everyone_targeted_usd": round(target_all, 2),
        },
        "churn_rate_by_contract": df.groupby("contract")[TARGET].mean().round(4).to_dict(),
        "churn_rate_by_tenure_band": df.groupby("tenure_band")[TARGET].mean().round(4).to_dict(),
    }
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lifts.round(4).to_csv(RESULTS / "lift_table.csv")

    # Figure 1: lift by decile.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(lifts.index, lifts["churn_rate"] * 100, color="#2b6cb0")
    ax.axhline(y_test.mean() * 100, color="black", ls="--", lw=1, label=f"Average churn {y_test.mean():.1%}")
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Actual churn rate (%)")
    ax.set_xticks(lifts.index)
    ax.legend(frameon=False)
    ax.set_title(f"Churn rate by predicted risk decile ({best}, test set)")
    fig.tight_layout()
    fig.savefig(FIGURES / "lift_by_decile.png", dpi=150)
    plt.close(fig)

    # Figure 2: campaign value against threshold.
    grid = np.arange(0.05, 0.96, 0.01)
    vals = [campaign_value(y_test.to_numpy(), proba, t, value, OFFER_COST, SAVE_RATE) for t in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, np.array(vals) / 1000, color="#2f855a")
    ax.axvline(thr, color="black", ls="--", lw=1, label=f"Best threshold {thr:.2f}")
    ax.set_xlabel("Model score threshold for sending an offer")
    ax.set_ylabel("Net campaign value (USD thousands)")
    ax.set_title(f"Retention campaign value on {len(y_test):,} test customers")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "campaign_value.png", dpi=150)
    plt.close(fig)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(candidate_models()[best].fit(X, y), ROOT / "models" / "churn_model.joblib")
    print(json.dumps({k: metrics[k] for k in ("best_model", "test_scores_at_0.5", "top_decile_lift", "campaign")}, indent=2))


if __name__ == "__main__":
    main()
