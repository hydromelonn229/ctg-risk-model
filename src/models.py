"""Leakage-safe baselines and models for CTG risk prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
TARGET_THRESHOLDS = (7.05, 7.15, 7.20)

FEATURE_COLUMNS = [
    "baseline_fhr_bpm",
    "fhr_std_bpm",
    "short_term_variability_bpm",
    "long_term_variability_bpm",
    "n_accelerations",
    "acceleration_duration_s",
    "n_decelerations",
    "deceleration_mean_depth_bpm",
    "deceleration_mean_duration_s",
    "n_contractions",
    "contraction_rate_per_10min",
    "n_late_decelerations",
    "pct_signal_missing",
    "pct_signal_loss_before_interpolation",
    "pct_signal_loss_after_interpolation",
    "longest_gap_before_interpolation_s",
    "longest_gap_after_interpolation_s",
    "interpolated_samples",
]


def rule_based_risk(X: pd.DataFrame) -> np.ndarray:
    """Simple high-sensitivity rule baseline using only CTG-derived fields."""
    risk = (
        (X["baseline_fhr_bpm"] < 110)
        | (X["baseline_fhr_bpm"] > 160)
        | (X["long_term_variability_bpm"] < 5)
        | (X["n_decelerations"] >= 3)
        | (X["n_late_decelerations"] >= 1)
    )
    return risk.astype(float).to_numpy()


def _metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    negatives = scores[y_true == 0]
    positives = scores[y_true == 1]
    threshold = float(np.quantile(negatives, 0.90)) if len(negatives) else np.inf
    predicted_at_specificity = scores >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_90pct_specificity": float(
            predicted_at_specificity[y_true == 1].mean()
        ) if len(positives) else np.nan,
        "brier": float(brier_score_loss(y_true, scores)),
    }


def _model_specs() -> dict[str, tuple[Pipeline, dict[str, list]]]:
    logistic_base = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest_base = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced",
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return {
        "logistic_l2": (
            logistic_base,
            {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "logistic_l1": (
            logistic_base,
            {"clf__penalty": ["l1"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            forest_base,
            {
                "clf__max_depth": [3, 6, None],
                "clf__min_samples_leaf": [2, 5],
            },
        ),
    }


def evaluate_threshold(
    feature_table: pd.DataFrame,
    threshold: float,
    n_splits: int = N_SPLITS,
    n_repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Evaluate baselines and models with repeated nested CV."""
    usable = feature_table.loc[~feature_table["excluded"]].copy()
    X = usable[FEATURE_COLUMNS]
    y = (usable["pH"] < threshold).astype(int).to_numpy()
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    rows: list[dict[str, float | str]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        prevalence_score = np.full(len(test_idx), y_train.mean(), dtype=float)
        metric_values = _metrics(y_test, prevalence_score)
        rows.append(
            {
                "threshold": threshold,
                "model": "prevalence_baseline",
                "fold": fold,
                **metric_values,
            }
        )

        rule_score = rule_based_risk(X_test)
        rows.append(
            {
                "threshold": threshold,
                "model": "rule_based_baseline",
                "fold": fold,
                **_metrics(y_test, rule_score),
            }
        )

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold)
        for model_name, (pipeline, grid) in _model_specs().items():
            search = GridSearchCV(
                pipeline,
                grid,
                scoring="average_precision",
                cv=inner,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            scores = search.predict_proba(X_test)[:, 1]
            rows.append(
                {
                    "threshold": threshold,
                    "model": model_name,
                    "fold": fold,
                    "best_params": str(search.best_params_),
                    **_metrics(y_test, scores),
                }
            )
    return pd.DataFrame(rows)


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated-CV metrics with empirical 95% intervals."""
    metric_names = ["roc_auc", "pr_auc", "recall_at_90pct_specificity", "brier"]
    summaries = []
    for (threshold, model), group in fold_results.groupby(["threshold", "model"]):
        row = {"threshold": threshold, "model": model, "folds": len(group)}
        for metric in metric_names:
            values = group[metric].dropna().to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025)) if len(values) else np.nan
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975)) if len(values) else np.nan
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values(["threshold", "model"])
