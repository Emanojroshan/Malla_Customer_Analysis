"""
Mall Customer Segmentation - Core Analysis Module
==================================================
Handles EDA, clustering, and model training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load and return the mall customer dataset."""
    df = pd.read_csv(filepath)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables and drop unused columns.
    Returns a cleaned DataFrame ready for ML.
    """
    df = df.copy()

    # Encode Gender
    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    # Drop ID column
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    # Drop non-numeric Visit_Frequency if datetime-like
    if df["Visit_Frequency"].dtype == object:
        df.drop(columns=["Visit_Frequency"], inplace=True)

    print("✅ Preprocessing complete")
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def eda_summary(df: pd.DataFrame) -> None:
    """Print basic EDA summary."""
    print("\n📊 Dataset Info:")
    print(df.info())
    print("\n📈 Descriptive Statistics:")
    print(df.describe().round(2))
    print("\n❓ Missing Values:")
    print(df.isnull().sum())
    print("\n👥 Gender Distribution:")
    print(df["Gender"].value_counts() if df["Gender"].dtype == object else "Already encoded")


def plot_distributions(df: pd.DataFrame) -> None:
    """Plot histograms and boxplots for key numeric features."""
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Annual_Spend"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, 4 * len(numeric_cols)))

    for i, col in enumerate(numeric_cols):
        # Histogram
        axes[i, 0].hist(df[col], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        axes[i, 0].set_title(f"{col} — Distribution", fontsize=12)
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Frequency")

        # Boxplot
        axes[i, 1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                            boxprops=dict(facecolor="steelblue", alpha=0.7))
        axes[i, 1].set_title(f"{col} — Boxplot", fontsize=12)

    plt.tight_layout()
    plt.savefig("static/eda_distributions.png", dpi=120)
    plt.show()
    print("✅ Distribution plots saved")


def plot_correlation(df: pd.DataFrame) -> None:
    """Plot a heatmap of feature correlations."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig("static/correlation_heatmap.png", dpi=120)
    plt.show()
    print("✅ Correlation heatmap saved")


# ─────────────────────────────────────────────
# 3. CLUSTERING
# ─────────────────────────────────────────────

def elbow_method(df: pd.DataFrame,
                 features: list = None,
                 max_k: int = 10) -> None:
    """Plot the Elbow Method to find optimal number of clusters."""
    if features is None:
        features = ["Annual Income (k$)", "Spending Score (1-100)"]

    X = df[features]
    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, "bo-", markersize=8)
    plt.axvline(x=5, color="red", linestyle="--", label="Optimal k=5")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("Elbow Method — Optimal k Selection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/elbow_plot.png", dpi=120)
    plt.show()
    print("✅ Elbow plot saved")


def run_kmeans(df: pd.DataFrame,
               features: list = None,
               n_clusters: int = 5) -> tuple[pd.DataFrame, KMeans]:
    """
    Fit K-Means and return DataFrame with Cluster labels + the fitted model.
    """
    if features is None:
        features = ["Annual Income (k$)", "Spending Score (1-100)"]

    X = df[features]
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = km.fit_predict(X)

    print(f"\n✅ K-Means complete — {n_clusters} clusters")
    print(df["Cluster"].value_counts().sort_index())
    return df, km


def plot_clusters(df: pd.DataFrame,
                  km: KMeans,
                  x_col: str = "Annual Income (k$)",
                  y_col: str = "Spending Score (1-100)") -> None:
    """Scatter plot of customer clusters with centroids."""
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(df[x_col], df[y_col],
                          c=df["Cluster"], cmap="tab10", alpha=0.6, s=30)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, c="black", marker="X", label="Centroids", zorder=5)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title("Customer Segments — K-Means Clustering", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/clusters.png", dpi=120)
    plt.show()
    print("✅ Cluster plot saved")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_random_forest(df: pd.DataFrame,
                        feature_cols: list = None) -> RandomForestClassifier:
    """
    Train a Random Forest to predict customer cluster.
    Returns the fitted classifier.
    """
    if feature_cols is None:
        feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]

    X = df[feature_cols]
    y = df["Cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    accuracy = rf.score(X_test, y_test)
    print(f"\n✅ Random Forest trained — Test Accuracy: {accuracy:.2%}")
    return rf


def train_regression_models(df: pd.DataFrame,
                             target: str = "Spending Score (1-100)") -> dict:
    """
    Train Linear Regression and Decision Tree for regression.
    Returns dict with models and MSE scores.
    """
    drop_cols = [target, "Cluster"] if "Cluster" in df.columns else [target]
    if "Visit_Frequency" in df.columns:
        drop_cols.append("Visit_Frequency")

    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    mse_lin = mean_squared_error(y_test, lin_reg.predict(X_test))

    tree_reg = DecisionTreeRegressor(random_state=42, max_depth=5)
    tree_reg.fit(X_train, y_train)
    mse_tree = mean_squared_error(y_test, tree_reg.predict(X_test))

    print(f"\n📉 Linear Regression MSE:    {mse_lin:.2f}")
    print(f"🌳 Decision Tree MSE:        {mse_tree:.2f}")

    return {
        "linear_regression": {"model": lin_reg, "mse": mse_lin},
        "decision_tree": {"model": tree_reg, "mse": mse_tree},
    }


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Full pipeline run
    df_raw = load_data("data/Mall_customers.csv")
    eda_summary(df_raw)

    df_clean = preprocess(df_raw)
    plot_distributions(df_raw)   # use raw for readability
    plot_correlation(df_clean)

    elbow_method(df_clean)
    df_clustered, kmeans_model = run_kmeans(df_clean)
    plot_clusters(df_clustered, kmeans_model)

    rf_model = train_random_forest(df_clustered)
    regression_results = train_regression_models(df_clustered)

    print("\n🎉 Full analysis pipeline complete!")
