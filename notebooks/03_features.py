"""Phase 3: extract interpretable CTG features from the cleaned final window."""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import build_metadata_table
from src.features import build_feature_table
from src.preprocessing import build_cleaned_metadata_table

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

metadata = build_metadata_table(RAW_DIR)
metadata = build_cleaned_metadata_table(RAW_DIR, metadata)
feature_table = build_feature_table(RAW_DIR, metadata)
feature_table.to_csv(PROCESSED_DIR / "ctg_features.csv", index=False)

print(f"Saved {len(feature_table)} feature rows.")
print("Feature columns:")
print(feature_table.columns.tolist())
print(feature_table.filter(regex="^(baseline|fhr|short|long|n_|acceleration|deceleration|contraction|pct_signal)").describe().T)
