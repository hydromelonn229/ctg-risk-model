"""Calibration and interpretable explanations for the CTG risk prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_loading import FS, load_record, resolve_data_dir
from src.features import extract_ctg_features
from src.models import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    _model_specs,
)
from src.preprocessing import prepare_recording

PRIMARY_THRESHOLD = 7.15
CASE_DECISION_THRESHOLD = 0.20


def _fit_logistic_l2(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Fit a final standardized logistic model without using held-out data."""
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X, y)
    return model


def _fit_calibrator(
    method: str, raw_train: np.ndarray, y_train: np.ndarray
) -> object:
    if method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(raw_train.reshape(-1, 1), y_train)
        return calibrator
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_train, y_train)
        return calibrator
    raise ValueError("method must be 'platt' or 'isotonic'")


def _apply_calibrator(calibrator: object, method: str, scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    return np.asarray(calibrator.predict(scores), dtype=float)


def _calibration_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, scores)),
        "mean_predicted_risk": float(np.mean(scores)),
        "observed_prevalence": float(np.mean(y)),
    }


def evaluate_calibration(
    feature_table: pd.DataFrame,
    threshold: float = PRIMARY_THRESHOLD,
    method: str = "platt",
    n_splits: int = 5,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate leakage-safe out-of-fold raw and calibrated predictions.

    Calibration is fitted only to inner out-of-fold predictions from each
    outer training partition. The outer test partition remains untouched.
    """
    usable = feature_table.loc[~feature_table["excluded"]].copy().reset_index(drop=True)
    X, y = usable[FEATURE_COLUMNS], (usable["pH"] < threshold).astype(int).to_numpy()
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    prediction_rows: list[dict[str, float | int | str]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        inner = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold
        )
        train_scores = np.full(len(train_idx), np.nan)
        model, grid = _model_specs()["logistic_l2"]
        # Select the regularization strength within the outer training partition.
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(model, grid, scoring="average_precision", cv=inner, refit=True)
        search.fit(X.iloc[train_idx], y[train_idx])
        outer_scores = search.predict_proba(X.iloc[test_idx])[:, 1]

        for cal_train, cal_valid in inner.split(X.iloc[train_idx], y[train_idx]):
            fold_model = GridSearchCV(
                model, grid, scoring="average_precision",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold + 1),
                refit=True,
            )
            fold_model.fit(
                X.iloc[train_idx].iloc[cal_train],
                y[train_idx][cal_train],
            )
            train_scores[cal_valid] = fold_model.predict_proba(
                X.iloc[train_idx].iloc[cal_valid]
            )[:, 1]
        calibrator = _fit_calibrator(method, train_scores, y[train_idx])
        calibrated_scores = _apply_calibrator(calibrator, method, outer_scores)
        for row_idx, score, calibrated in zip(test_idx, outer_scores, calibrated_scores):
            prediction_rows.append(
                {
                    "record": usable.loc[row_idx, "record"],
                    "threshold": threshold,
                    "fold": fold,
                    "y_true": int(y[row_idx]),
                    "raw_score": float(score),
                    "calibrated_score": float(calibrated),
                }
            )

    predictions = pd.DataFrame(prediction_rows).sort_values("record")
    metric_rows = []
    for score_name in ("raw_score", "calibrated_score"):
        scores = predictions[score_name].to_numpy()
        observed = predictions["y_true"].to_numpy()
        metric_rows.append(
            {"threshold": threshold, "method": score_name, **_calibration_metrics(observed, scores)}
        )
    return predictions, pd.DataFrame(metric_rows)


def standardized_logistic_coefficients(
    feature_table: pd.DataFrame, threshold: float = PRIMARY_THRESHOLD
) -> pd.DataFrame:
    """Return final-model coefficients in standardized feature units."""
    usable = feature_table.loc[~feature_table["excluded"]].copy()
    X, y = usable[FEATURE_COLUMNS], (usable["pH"] < threshold).astype(int).to_numpy()
    model = _fit_logistic_l2(X, y)
    coefficients = model.named_steps["clf"].coef_[0]
    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
            "direction": np.where(coefficients >= 0, "higher risk", "lower risk"),
        }
    )
    return result.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def feature_stability(
    feature_table: pd.DataFrame,
    threshold: float = PRIMARY_THRESHOLD,
    top_k: int = 5,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure how often each feature appears in the top-k by coefficient size."""
    usable = feature_table.loc[~feature_table["excluded"]].copy()
    X, y = usable[FEATURE_COLUMNS], (usable["pH"] < threshold).astype(int).to_numpy()
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE
    )
    rows = []
    for fold, (train_idx, _) in enumerate(outer.split(X, y)):
        model = _fit_logistic_l2(X.iloc[train_idx], y[train_idx])
        coefficients = model.named_steps["clf"].coef_[0]
        order = np.argsort(np.abs(coefficients))[::-1][:top_k]
        for rank, index in enumerate(order, start=1):
            rows.append({"fold": fold, "feature": FEATURE_COLUMNS[index], "rank": rank})
    ranks = pd.DataFrame(rows)
    summary = (
        ranks.groupby("feature")
        .agg(
            top_k_count=("feature", "size"),
            median_rank=("rank", "median"),
        )
        .assign(top_k_fraction=lambda frame: frame["top_k_count"] / (n_splits * n_repeats))
        .sort_values(["top_k_count", "median_rank"], ascending=[False, True])
        .reset_index()
    )
    return ranks, summary


def choose_case_studies(
    feature_table: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float = PRIMARY_THRESHOLD,
) -> pd.DataFrame:
    """Choose representative TP, FP, and FN records from OOF predictions."""
    usable = feature_table.loc[~feature_table["excluded"]].copy()
    prediction_summary = (
        predictions.groupby("record", as_index=False)
        .agg(
            threshold=("threshold", "first"),
            y_true=("y_true", "first"),
            raw_score=("raw_score", "mean"),
            calibrated_score=("calibrated_score", "mean"),
        )
    )
    merged = usable[["record", "pH"] + FEATURE_COLUMNS].merge(
        prediction_summary, on="record", validate="one_to_one"
    )
    # Use a deliberately documented exploratory operating point near the
    # primary prevalence; this is not a clinically validated threshold.
    merged["predicted"] = (
        merged["calibrated_score"] >= CASE_DECISION_THRESHOLD
    ).astype(int)
    selections = []
    for case_type, mask in {
        "true_positive": (merged["y_true"] == 1) & (merged["predicted"] == 1),
        "false_positive": (merged["y_true"] == 0) & (merged["predicted"] == 1),
        "false_negative": (merged["y_true"] == 1) & (merged["predicted"] == 0),
    }.items():
        candidates = merged.loc[mask].copy()
        if candidates.empty:
            continue
        target = 0.75 if case_type != "false_negative" else 0.25
        candidates["distance"] = np.abs(candidates["calibrated_score"] - target)
        row = candidates.sort_values("distance").iloc[0].to_dict()
        row["case_type"] = case_type
        selections.append(row)
    return pd.DataFrame(selections)


def local_contributions(
    case_studies: pd.DataFrame,
    feature_table: pd.DataFrame,
    threshold: float = PRIMARY_THRESHOLD,
) -> pd.DataFrame:
    """Calculate standardized logistic feature contributions for selected cases."""
    usable = feature_table.loc[~feature_table["excluded"]].copy()
    X, y = usable[FEATURE_COLUMNS], (usable["pH"] < threshold).astype(int).to_numpy()
    model = _fit_logistic_l2(X, y)
    imputed = model.named_steps["impute"].transform(case_studies[FEATURE_COLUMNS])
    scaled = model.named_steps["scale"].transform(imputed)
    coefficients = model.named_steps["clf"].coef_[0]
    rows = []
    for case_idx, (_, case) in enumerate(case_studies.iterrows()):
        contributions = scaled[case_idx] * coefficients
        order = np.argsort(np.abs(contributions))[::-1][:5]
        for rank, feature_idx in enumerate(order, start=1):
            rows.append(
                {
                    "case_type": case["case_type"],
                    "record": case["record"],
                    "rank": rank,
                    "feature": FEATURE_COLUMNS[feature_idx],
                    "contribution": float(contributions[feature_idx]),
                    "feature_value": float(case[FEATURE_COLUMNS[feature_idx]]),
                }
            )
    return pd.DataFrame(rows)


def plot_calibration(
    predictions: pd.DataFrame,
    output_path: Path,
    n_bins: int = 8,
) -> None:
    """Save a reliability diagram for raw and calibrated OOF scores."""
    y = predictions["y_true"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for column, label in (
        ("raw_score", "Raw logistic score"),
        ("calibrated_score", "Platt-calibrated score"),
    ):
        observed, predicted = calibration_curve(y, predictions[column], n_bins=n_bins, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed event frequency", title="pH < 7.15 calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_case_trace(
    case: pd.Series,
    raw_dir: Path,
    output_path: Path,
) -> None:
    """Save the final-window FHR/UC trace for one selected case."""
    fhr, uc, _ = prepare_recording(str(case["record"]), resolve_data_dir(raw_dir), fs=FS)
    time_min = np.arange(len(fhr)) / FS / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time_min, fhr, linewidth=0.7)
    axes[0].axhline(110, color="grey", linestyle="--", linewidth=0.7)
    axes[0].axhline(160, color="grey", linestyle="--", linewidth=0.7)
    axes[0].set_ylabel("FHR (bpm)")
    axes[1].plot(time_min, uc, linewidth=0.7, color="tab:orange")
    axes[1].set_ylabel("UC")
    axes[1].set_xlabel("Minutes before delivery-window end")
    fig.suptitle(
        f"{case['case_type'].replace('_', ' ').title()} — {case['record']} "
        f"(pH {case['pH']:.2f}, score {case['calibrated_score']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
