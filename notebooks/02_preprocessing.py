"""Phase 2: clean FHR signals and prepare the frozen analysis window."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow this script to run directly from the repository root or notebooks folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading import build_metadata_table
from src.preprocessing import (
    ANALYSIS_WINDOW_MIN,
    MAX_INTERPOLATION_GAP_S,
    build_cleaned_metadata_table,
    prepare_recording,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

metadata = build_metadata_table(RAW_DIR)
cleaned_metadata = build_cleaned_metadata_table(
    RAW_DIR,
    metadata,
    window_min=ANALYSIS_WINDOW_MIN,
    max_gap_s=MAX_INTERPOLATION_GAP_S,
)
cleaned_metadata.to_csv(PROCESSED_DIR / "metadata_preprocessed.csv", index=False)

print(f"Analysis window: final {ANALYSIS_WINDOW_MIN:g} minutes")
print(f"Interpolation limit: {MAX_INTERPOLATION_GAP_S:g} seconds")
print(f"Recordings: {len(cleaned_metadata)}")
print(
    "Excluded after Phase 2 rules: "
    f"{int(cleaned_metadata['excluded'].sum())}"
)
print(
    "Final analysis set: "
    f"{int((~cleaned_metadata['excluded']).sum())}"
)
print(
    cleaned_metadata[
        [
            "pct_signal_loss_before_interpolation",
            "pct_signal_loss_after_interpolation",
            "longest_gap_before_interpolation_s",
            "interpolated_samples",
        ]
    ].describe().to_string()
)

# Show representative traces with different signal-quality profiles.
selected = (
    cleaned_metadata.sort_values(
        "pct_signal_loss_before_interpolation"
    )["record"]
    .iloc[[0, len(cleaned_metadata) // 2, -1]]
    .tolist()
)
fig, axes = plt.subplots(len(selected), 1, figsize=(14, 9), sharex=True)
axes = np.atleast_1d(axes)
time_min = np.arange(int(ANALYSIS_WINDOW_MIN * 60 * 4)) / 4 / 60

for ax, record_name in zip(axes, selected):
    raw_window, _, _ = prepare_recording(
        record_name,
        RAW_DIR,
        window_min=ANALYSIS_WINDOW_MIN,
        max_gap_s=0,
    )
    cleaned_window, _, quality = prepare_recording(
        record_name,
        RAW_DIR,
        window_min=ANALYSIS_WINDOW_MIN,
        max_gap_s=MAX_INTERPOLATION_GAP_S,
    )
    ax.plot(time_min, raw_window, color="#999999", linewidth=0.6, label="Raw valid FHR")
    ax.plot(time_min, cleaned_window, color="#1f77b4", linewidth=0.8, label="Cleaned FHR")
    ax.set_ylim(40, 230)
    ax.set_ylabel("FHR (bpm)")
    ax.set_title(
        f"Record {record_name}: "
        f"{quality['pct_signal_loss_before_interpolation']:.1f}% invalid before cleaning"
    )
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Minutes before delivery window start")
fig.suptitle("Phase 2: final 30-minute FHR cleaning", y=1.01)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_fhr_cleaning_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
