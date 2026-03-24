"""
Mall Customer Segmentation - Prediction Helper
===============================================
Loads trained models and predicts cluster for new customers.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Cluster labels for human-readable output
CLUSTER_PROFILES = {
    0: {"label": "Budget Shoppers",        "desc": "Low income, low spending. Price-sensitive."},
    1: {"label": "Careful Spenders",       "desc": "High income, but low spending. Needs engagement."},
    2: {"label": "Balanced Customers",     "desc": "Moderate income and spending. Core audience."},
    3: {"label": "Impulsive Spenders",     "desc": "Lower income but high spending score. Trend-driven."},
    4: {"label": "Premium Customers ⭐",   "desc": "High income, high spending. Most valuable segment."},
}


def predict_cluster(model: RandomForestClassifier,
                    annual_income: float,
                    spending_score: float) -> dict:
    """
    Predict the cluster for a single customer.

    Parameters
    ----------
    model : fitted RandomForestClassifier
    annual_income : float — customer's annual income in k$
    spending_score : float — mall spending score (1–100)

    Returns
    -------
    dict with cluster id, label, and description
    """
    features = pd.DataFrame(
        [[annual_income, spending_score]],
        columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )
    cluster_id = int(model.predict(features)[0])
    profile = CLUSTER_PROFILES.get(cluster_id, {
        "label": f"Cluster {cluster_id}", "desc": "No profile available."
    })

    return {
        "cluster": cluster_id,
        "label": profile["label"],
        "description": profile["desc"],
        "inputs": {
            "annual_income_k$": annual_income,
            "spending_score": spending_score,
        },
    }


def batch_predict(model: RandomForestClassifier,
                  df: pd.DataFrame,
                  income_col: str = "Annual Income (k$)",
                  score_col: str = "Spending Score (1-100)") -> pd.DataFrame:
    """
    Predict clusters for all rows in a DataFrame.

    Returns the DataFrame with added 'Cluster' and 'Segment_Label' columns.
    """
    X = df[[income_col, score_col]]
    df = df.copy()
    df["Cluster"] = model.predict(X)
    df["Segment_Label"] = df["Cluster"].map(
        lambda c: CLUSTER_PROFILES.get(c, {}).get("label", f"Cluster {c}")
    )
    return df
