"""Regenerate every figure and metric used in the deployed research paper.

Run from the repo root:

    python scripts/build_paper_assets.py

Outputs:
    docs/assets/fig_before_after.png
    docs/assets/fig_model_vs_baseline.png
    docs/assets/fig_leakage.png
    docs/assets/fig_feature_importance.png
    docs/assets/metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ml_utils import RAW_PATH, ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

ASSET_DIR = ROOT / "docs" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NUMERIC = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
    "days_with_impressions", "days_with_sessions", "content_age_days",
    "days_since_last_update", "ctr", "avg_position", "engagement_rate",
    "scroll_rate", "ai_traffic_pct",
]
CATEG = [
    "competition_level", "content_type", "main_intent", "age_tier",
    "freshness_tier", "word_count_tier", "impression_tier", "position_tier",
]

INK = "#1f2937"
MUTED = "#94a3b8"
GRID = "#e5e7eb"
PRIMARY = "#0f766e"
WARN = "#b45309"
ACCENT = "#c2410c"


def build_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(RAW_PATH)
    df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)
    df["log_impressions_90d"] = np.log1p(df["impressions_90d"])
    df["log_clicks_90d"] = np.log1p(df["clicks_90d"])
    df["log_sessions_90d"] = np.log1p(df["sessions_90d"])
    df["log_ai_sessions_90d"] = np.log1p(df["ai_sessions_90d"])

    num = df[NUMERIC].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    cat = df[CATEG].fillna("unknown").astype(str)
    X = pd.concat(
        [num.reset_index(drop=True), pd.get_dummies(cat, prefix=CATEG, dtype=float).reset_index(drop=True)],
        axis=1,
    )
    y = df["is_declining_label"].astype(int)
    return df, X, y


def precision_at_k(y_true, scores, k: int = 50) -> float:
    frame = pd.DataFrame({"y": list(y_true), "s": list(scores)})
    top = frame.sort_values("s", ascending=False).head(min(k, len(frame)))
    return float(top["y"].mean()) if len(top) else 0.0


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        class_weight="balanced_subsample",
        max_depth=10,
        min_samples_leaf=25,
        n_estimators=200,
        n_jobs=-1,
        random_state=seed,
    )


def score(y_true, prob) -> dict:
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_at_50": round(precision_at_k(y_true, prob), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "avg_precision": round(float(average_precision_score(y_true, prob)), 4),
    }


def holdout_indices(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    clients = df["client_id"].astype(str).unique()
    shuffled = rng.permutation(clients)
    test_clients = set(shuffled[: max(1, round(len(shuffled) * 0.2))])
    mask = df["client_id"].astype(str).isin(test_clients).to_numpy()
    return np.where(~mask)[0], np.where(mask)[0], len(clients), len(test_clients)


def bar_axes(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=15, color=INK, pad=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.tick_params(colors="#334155", labelsize=12)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.savefig(ASSET_DIR / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    df, X, y = build_frame()
    train_idx, test_idx, n_clients, n_test_clients = holdout_indices(df, RANDOM_STATE)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # --- baseline + three learned models on the honest client-holdout split ---
    stale = (df["days_since_last_update"] >= 90).astype(float)
    visible = (df["impressions_prev_30d"] >= df["impressions_prev_30d"].median()).astype(float)
    rule_score = (stale * visible * df["impressions_prev_30d"]).to_numpy()

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE
        ),
        "random_forest": make_rf(RANDOM_STATE),
    }

    results = {}
    results["dummy_majority"] = score(y_test, np.ones(len(y_test)))
    results["rule_baseline"] = score(y_test, rule_score[test_idx])
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = score(y_test, model.predict_proba(X_test)[:, 1])

    rf = models["random_forest"]
    importance = (
        pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(8)
    )

    # --- BEFORE (random rows) vs AFTER (client-holdout) ---
    tr_r, te_r = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    rf_before = make_rf(RANDOM_STATE)
    rf_before.fit(X.iloc[tr_r], y.iloc[tr_r])
    before = score(y.iloc[te_r], rf_before.predict_proba(X.iloc[te_r])[:, 1])

    rf_after = make_rf(RANDOM_STATE)
    rf_after.fit(X_train, y_train)
    after = score(y_test, rf_after.predict_proba(X_test)[:, 1])

    # --- leakage confessions (same split, add one suspect column) ---
    leak_rows = [{"feature_set": "honest (used)", **after}]
    for label, column in [
        ("+ impressions_last_30d (window overlap)", "impressions_last_30d"),
        ("+ trend_pct (label-derived)", "trend_pct"),
    ]:
        X_leak = X.copy()
        X_leak[f"{column}_suspect"] = df[column].fillna(0).to_numpy()
        rf_leak = make_rf(RANDOM_STATE)
        rf_leak.fit(X_leak.iloc[train_idx], y_train)
        leak_rows.append(
            {"feature_set": label, **score(y_test, rf_leak.predict_proba(X_leak.iloc[test_idx])[:, 1])}
        )

    # --- where the honest model still struggles (fixed holdout, seed 42) ---
    holdout_df = df.iloc[test_idx].copy()
    holdout_df["prob"] = rf_after.predict_proba(X_test)[:, 1]
    holdout_df["correct"] = ((holdout_df["prob"] >= 0.5).astype(int) == y_test.to_numpy()).astype(int)
    by_type = (
        holdout_df.groupby("content_type")["correct"]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
    )
    by_position = (
        holdout_df.groupby("position_tier")["correct"]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
    )
    error_analysis = {
        "by_content_type": [
            {"group": str(r.content_type), "accuracy": round(float(r.mean), 4), "n": int(r.count)}
            for r in by_type.itertuples()
        ],
        "by_position_tier": [
            {"group": str(r.position_tier), "accuracy": round(float(r.mean), 4), "n": int(r.count)}
            for r in by_position.itertuples()
        ],
    }

    # --- stability: repeat split + model across 5 seeds ---
    stability = []
    for seed in [42, 7, 13, 2024, 99]:
        tr, te, _, _ = holdout_indices(df, seed)
        rf_s = make_rf(seed)
        rf_s.fit(X.iloc[tr], y.iloc[tr])
        stability.append(score(y.iloc[te], rf_s.predict_proba(X.iloc[te])[:, 1]))
    stab = pd.DataFrame(stability)
    stability_summary = {
        "seeds": [42, 7, 13, 2024, 99],
        "precision_at_50_mean": round(float(stab["precision_at_50"].mean()), 4),
        "precision_at_50_sd": round(float(stab["precision_at_50"].std(ddof=1)), 4),
        "roc_auc_mean": round(float(stab["roc_auc"].mean()), 4),
        "roc_auc_sd": round(float(stab["roc_auc"].std(ddof=1)), 4),
    }

    raw_columns = int(pd.read_csv(RAW_PATH, nrows=0).shape[1])

    payload = {
        "rows": int(len(df)),
        "columns": raw_columns,
        "clients": int(df["client_id"].nunique()),
        "base_rate_overall": round(float(y.mean()), 4),
        "label_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "features": int(X.shape[1]),
        "numeric_features": len(NUMERIC),
        "categorical_source_columns": len(CATEG),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "clients_total": int(n_clients),
        "clients_held_out": int(n_test_clients),
        "train_base_rate": round(float(y_train.mean()), 4),
        "test_base_rate": round(float(y_test.mean()), 4),
        "random_split": {**before, "test_base_rate": round(float(y.iloc[te_r].mean()), 4)},
        "client_holdout": {**after, "test_base_rate": round(float(y_test.mean()), 4)},
        "models_client_holdout": results,
        "leakage": leak_rows,
        "stability_5_seeds": stability_summary,
        "error_analysis": error_analysis,
        "top_features": [
            {"feature": row.feature, "importance": round(float(row.importance), 4)}
            for row in importance.itertuples()
        ],
    }
    (ASSET_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------- Figure 1: before vs after ----------------
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    metrics = ["precision_at_50", "roc_auc"]
    labels = ["Precision@50", "ROC-AUC"]
    xpos = np.arange(len(metrics))
    width = 0.34
    before_vals = [before[m] for m in metrics]
    after_vals = [after[m] for m in metrics]
    b1 = ax.bar(xpos - width / 2, before_vals, width, color=MUTED, label="Random row split (before)")
    b2 = ax.bar(xpos + width / 2, after_vals, width, color=PRIMARY, label="Client-holdout (after)")
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(
                f"{rect.get_height():.2f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=11, color=INK, fontweight="bold",
            )
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    bar_axes(ax, "Same model, honest split: precision drops, ranking holds")
    ax.legend(frameon=False, loc="upper right", fontsize=11)
    save(fig, "fig_before_after.png")

    # ---------------- Figure 2: model vs baselines ----------------
    order = ["dummy_majority", "rule_baseline", "logistic_regression", "decision_tree", "random_forest"]
    pretty = ["All-positive\nbaseline", "Transparent\nrule", "Logistic\nregression", "Decision\ntree", "Random\nforest"]
    vals = [results[k]["precision_at_50"] for k in order]
    colors = [MUTED, MUTED, "#64748b", "#475569", PRIMARY]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.barh(pretty, vals, color=colors, height=0.62)
    for rect, val in zip(bars, vals):
        ax.annotate(
            f"{val:.2f}", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
            textcoords="offset points", xytext=(7, 0), va="center",
            fontsize=11, color=INK, fontweight="bold",
        )
    ax.axvline(
        y_test.mean(), color=ACCENT, linestyle="--", linewidth=1.4,
        label=f"held-out base rate ({y_test.mean():.2f})",
    )
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Precision@50 (share of top-50 flagged pages that truly declined)", fontsize=11, color="#334155")
    bar_axes(ax, "Random forest triages best on unseen clients")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper right", fontsize=10.5)
    save(fig, "fig_model_vs_baseline.png")

    # ---------------- Figure 3: leakage confession ----------------
    names = ["Honest\nfeature set", "+ last-30d\n(window overlap)", "+ trend_pct\n(label-derived)"]
    leak_vals = [row["accuracy"] for row in leak_rows]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    colors = [PRIMARY, WARN, "#991b1b"]
    bars = ax.bar(names, leak_vals, color=colors, width=0.58)
    for rect, val in zip(bars, leak_vals):
        ax.annotate(
            f"{val:.3f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points", xytext=(0, 4), ha="center",
            fontsize=12, color=INK, fontweight="bold",
        )
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Test accuracy", fontsize=11, color="#334155")
    bar_axes(ax, "Adding one label-derived column fakes a perfect model")
    ax.text(
        0.02, 0.86, "0.999 = red flag, not progress",
        transform=ax.transAxes, fontsize=10.5, color="#991b1b", fontweight="bold",
    )
    save(fig, "fig_leakage.png")

    # ---------------- Figure 4: feature importance ----------------
    imp = importance.iloc[::-1]
    friendly = {
        "days_with_impressions": "Days with impressions (90d)",
        "log_impressions_90d": "Impressions (90d, log)",
        "avg_position": "Average position",
        "content_age_days": "Content age (days)",
        "log_clicks_90d": "Clicks (90d, log)",
        "char_count": "Character count",
        "age_tier_365+": "Age tier: 365+ days",
        "word_count": "Word count",
        "ctr": "CTR",
    }
    names = [friendly.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.barh(names, imp["importance"].to_numpy(), color="#0e7490", height=0.62)
    for rect, val in zip(bars, imp["importance"].to_numpy()):
        ax.annotate(
            f"{val:.3f}", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
            textcoords="offset points", xytext=(6, 0), va="center",
            fontsize=10, color=INK,
        )
    ax.set_xlabel("Random-forest importance (honest feature set)", fontsize=11, color="#334155")
    bar_axes(ax, "It leans on visibility, position, and age")
    ax.set_xlim(0, imp["importance"].max() * 1.18)
    save(fig, "fig_feature_importance.png")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
