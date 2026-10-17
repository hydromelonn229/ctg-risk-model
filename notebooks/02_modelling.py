"""Phase 4: leakage-safe baselines and repeated nested cross-validation."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import evaluate_threshold, summarize_results, TARGET_THRESHOLDS

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
feature_table = pd.read_csv(PROCESSED_DIR / "ctg_features.csv")

all_folds = []
for threshold in TARGET_THRESHOLDS:
    print(f"Evaluating pH < {threshold:.2f} ...")
    all_folds.append(evaluate_threshold(feature_table, threshold))

fold_results = pd.concat(all_folds, ignore_index=True)
summary = summarize_results(fold_results)
fold_results.to_csv(PROCESSED_DIR / "model_fold_results.csv", index=False)
summary.to_csv(PROCESSED_DIR / "model_results.csv", index=False)

print(summary.to_string(index=False))
