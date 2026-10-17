"""Run Phase 5 calibration and explainability analysis."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loading import RAW_DIR
from src.explainability import (
    choose_case_studies,
    evaluate_calibration,
    feature_stability,
    local_contributions,
    plot_calibration,
    plot_case_trace,
    standardized_logistic_coefficients,
)


PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
FEATURE_TABLE = pd.read_csv(
    PROCESSED / "ctg_features.csv",
    converters={"record": lambda value: str(value).split(".")[0]},
)

predictions, calibration_metrics = evaluate_calibration(FEATURE_TABLE)
predictions.to_csv(PROCESSED / "calibration_oof_predictions.csv", index=False)
calibration_metrics.to_csv(PROCESSED / "calibration_metrics.csv", index=False)
plot_calibration(predictions, FIGURES / "05_calibration.png")

coefficients = standardized_logistic_coefficients(FEATURE_TABLE)
coefficients.to_csv(PROCESSED / "logistic_coefficients.csv", index=False)

stability_ranks, stability_summary = feature_stability(FEATURE_TABLE)
stability_ranks.to_csv(PROCESSED / "feature_stability_by_fold.csv", index=False)
stability_summary.to_csv(PROCESSED / "feature_stability_summary.csv", index=False)

cases = choose_case_studies(FEATURE_TABLE, predictions)
cases.to_csv(PROCESSED / "case_studies.csv", index=False)
contributions = local_contributions(cases, FEATURE_TABLE)
contributions.to_csv(PROCESSED / "case_study_contributions.csv", index=False)

for _, case in cases.iterrows():
    plot_case_trace(
        case,
        ROOT / RAW_DIR,
        FIGURES / f"05_case_{case['case_type']}_{case['record']}.png",
    )

print("Calibration metrics:")
print(calibration_metrics.to_string(index=False))
print("\nTop standardized coefficients:")
print(coefficients.head(10).to_string(index=False))
print("\nFeature stability:")
print(stability_summary.to_string(index=False))
print("\nCase studies:")
print(cases[["case_type", "record", "pH", "y_true", "calibrated_score"]].to_string(index=False))
