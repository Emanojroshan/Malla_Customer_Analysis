"""
Mall Customer Segmentation — Flask Web App
==========================================
Trains the model on startup and serves predictions via web interface.
"""

import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

from src.analysis import load_data, preprocess, run_kmeans, train_random_forest
from src.predict import predict_cluster

# ─── App Setup ────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── Train on Startup ─────────────────────────────
DATA_PATH = os.path.join("data", "Mall_customers.csv")
df_raw = load_data(DATA_PATH)
df_clean = preprocess(df_raw)
df_clustered, kmeans_model = run_kmeans(df_clean)
rf_model = train_random_forest(df_clustered)

print("✅ Models ready — Flask server starting...")


# ─── Routes ───────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accepts JSON: { "annual_income": float, "spending_score": float }
    Returns predicted cluster info.
    """
    data = request.get_json()

    try:
        annual_income = float(data["annual_income"])
        spending_score = float(data["spending_score"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    result = predict_cluster(rf_model, annual_income, spending_score)
    return jsonify(result)


# ─── Entry Point ──────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
