"""
Credit Risk Scoring — Model Training & Evaluation
Trains Logistic Regression, Random Forest, and XGBoost.
Evaluates with AUC-ROC, PR-AUC, Gini, and business-cost metrics.
Explains predictions with SHAP.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier


# ─── Model Definitions ────────────────────────────────────────────────────────

def build_logistic(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(
        C=0.1, max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=seed,
    )


def build_random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )


def build_xgboost(scale_pos_weight: float = 3.5, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", use_label_encoder=False,
        random_state=seed, n_jobs=-1,
    )


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str = "") -> dict:
    """Return a dict of key metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    gini = 2 * auc - 1

    return {
        "model": model_name,
        "roc_auc": round(auc, 4),
        "pr_auc": round(pr_auc, 4),
        "gini": round(gini, 4),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def business_cost_analysis(
    y_true: pd.Series,
    y_prob: np.ndarray,
    avg_loan: float = 50_000,
    loss_given_default: float = 0.45,
    intervention_cost: float = 200,
) -> pd.DataFrame:
    """
    Sweep probability thresholds and compute net cost saved vs doing nothing.

    Cost model:
      - False Negative (missed default): avg_loan * loss_given_default
      - False Positive (wrongly flagged): intervention_cost
      - True Positive (caught default):  0 (prevented loss - intervention_cost)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    rows = []
    total_defaults = y_true.sum()
    baseline_loss = total_defaults * avg_loan * loss_given_default

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        cost_fn = fn * avg_loan * loss_given_default
        cost_fp = fp * intervention_cost
        cost_tp = tp * intervention_cost  # still costs something to intervene
        total_cost = cost_fn + cost_fp + cost_tp
        savings = baseline_loss - total_cost

        rows.append({
            "threshold": round(t, 2),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "total_cost_usd": total_cost,
            "net_savings_usd": savings,
        })

    return pd.DataFrame(rows)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_roc_curves(results: list[dict], y_test: pd.Series) -> plt.Figure:
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3B82F6", "#10B981", "#F59E0B"]

    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f"{res['model']} (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Credit Default Models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_precision_recall(results: list[dict], y_test: pd.Series) -> plt.Figure:
    """Overlay Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#3B82F6", "#10B981", "#F59E0B"]

    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, lw=2, color=color,
                label=f"{res['model']} (PR-AUC = {res['pr_auc']:.3f})")

    baseline = y_test.mean()
    ax.axhline(baseline, ls="--", color="gray", label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Credit Default Models")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shap_summary(model, X_test: pd.DataFrame, max_display: int = 20) -> plt.Figure:
    """SHAP beeswarm summary plot for XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test, max_display=max_display,
        show=False, plot_size=None,
    )
    plt.title("SHAP Feature Importance — XGBoost Credit Risk Model")
    plt.tight_layout()
    return plt.gcf()


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not Default", "Default"],
                yticklabels=["Not Default", "Default"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    return fig
