"""Signal cleaning and analysis-window preparation for CTU-UHB CTG records."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loading import FS, get_record_names, load_record, resolve_data_dir

FHR_MIN_BPM = 50.0
FHR_MAX_BPM = 220.0
ANALYSIS_WINDOW_MIN = 30.0
MAX_INTERPOLATION_GAP_S = 15.0
MAX_MISSING_FRACTION = 0.50
MIN_RECORDING_MIN = 10.0


def mark_invalid_fhr(
    fhr: np.ndarray,
    min_bpm: float = FHR_MIN_BPM,
    max_bpm: float = FHR_MAX_BPM,
) -> np.ndarray:
    """Return a copy with implausible FHR values represented as NaN."""
    values = np.asarray(fhr, dtype=float).copy()
    values[(values < min_bpm) | (values > max_bpm)] = np.nan
    return values


def _nan_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open index ranges for consecutive NaN runs."""
    missing = np.isnan(values)
    changes = np.diff(np.r_[False, missing, False].astype(int))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def interpolate_short_gaps(
    fhr: np.ndarray,
    fs: float = FS,
    max_gap_s: float = MAX_INTERPOLATION_GAP_S,
) -> np.ndarray:
    """Interpolate internal NaN gaps no longer than ``max_gap_s``.

    Leading/trailing gaps and longer gaps remain NaN. Interpolation is only
    performed between two observed values, so it cannot invent a signal at
    either edge of a recording.
    """
    values = np.asarray(fhr, dtype=float).copy()
    max_gap_samples = int(round(max_gap_s * fs))
    for start, end in _nan_runs(values):
        gap_length = end - start
        if gap_length > max_gap_samples:
            continue
        if start == 0 or end == len(values):
            continue
        if np.isnan(values[start - 1]) or np.isnan(values[end]):
            continue
        values[start:end] = np.linspace(
            values[start - 1],
            values[end],
            gap_length + 2,
        )[1:-1]
    return values


def longest_gap_seconds(fhr: np.ndarray, fs: float = FS) -> float:
    """Return the longest consecutive NaN run in seconds."""
    runs = _nan_runs(np.asarray(fhr, dtype=float))
    if not runs:
        return 0.0
    return max(end - start for start, end in runs) / fs


def select_analysis_window(
    signal: np.ndarray,
    fs: float = FS,
    window_min: float = ANALYSIS_WINDOW_MIN,
) -> np.ndarray:
    """Return the final fixed-duration window before delivery."""
    values = np.asarray(signal)
    window_samples = int(round(window_min * 60 * fs))
    if len(values) < window_samples:
        raise ValueError(
            f"Signal has {len(values)} samples but needs {window_samples} "
            f"for a {window_min:g}-minute window."
        )
    return values[-window_samples:]


def clean_fhr(
    fhr: np.ndarray,
    fs: float = FS,
    max_gap_s: float = MAX_INTERPOLATION_GAP_S,
) -> tuple[np.ndarray, dict[str, float]]:
    """Clean FHR and return the cleaned signal plus quality measurements."""
    invalid = mark_invalid_fhr(fhr)
    original_missing_fraction = float(np.isnan(invalid).mean())
    cleaned = interpolate_short_gaps(invalid, fs=fs, max_gap_s=max_gap_s)
    remaining_missing_fraction = float(np.isnan(cleaned).mean())
    quality = {
        "pct_signal_loss_before_interpolation": original_missing_fraction * 100,
        "pct_signal_loss_after_interpolation": remaining_missing_fraction * 100,
        "longest_gap_before_interpolation_s": longest_gap_seconds(invalid, fs),
        "longest_gap_after_interpolation_s": longest_gap_seconds(cleaned, fs),
        "interpolated_samples": float(
            np.count_nonzero(np.isnan(invalid) & ~np.isnan(cleaned))
        ),
    }
    return cleaned, quality


def prepare_recording(
    record_name: str,
    raw_dir: Path,
    fs: float = FS,
    window_min: float = ANALYSIS_WINDOW_MIN,
    max_gap_s: float = MAX_INTERPOLATION_GAP_S,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Load, window, clean, and measure one recording."""
    _, fhr, uc = load_record(record_name, raw_dir)
    fhr_window = select_analysis_window(fhr, fs=fs, window_min=window_min)
    uc_window = select_analysis_window(uc, fs=fs, window_min=window_min)
    cleaned_fhr, quality = clean_fhr(fhr_window, fs=fs, max_gap_s=max_gap_s)
    quality.update(
        {
            "record": record_name,
            "window_min": window_min,
            "window_samples": len(fhr_window),
            "excluded_excessive_signal_loss": (
                quality["pct_signal_loss_before_interpolation"]
                > MAX_MISSING_FRACTION * 100
            ),
        }
    )
    return cleaned_fhr, uc_window, quality


def build_cleaned_metadata_table(
    raw_dir: Path,
    metadata: pd.DataFrame,
    fs: float = FS,
    window_min: float = ANALYSIS_WINDOW_MIN,
    max_gap_s: float = MAX_INTERPOLATION_GAP_S,
) -> pd.DataFrame:
    """Add final-window signal-quality fields to the metadata table."""
    raw_dir = resolve_data_dir(raw_dir)
    quality_rows = []
    for record_name in get_record_names(raw_dir):
        _, _, quality = prepare_recording(
            record_name,
            raw_dir,
            fs=fs,
            window_min=window_min,
            max_gap_s=max_gap_s,
        )
        quality_rows.append(quality)
    quality_df = pd.DataFrame(quality_rows)
    result = metadata.merge(quality_df, on="record", how="inner", validate="one_to_one")
    result["excluded"] = (
        result["pH"].isna()
        | result["excluded_excessive_signal_loss"]
        | (result["duration_min"] < MIN_RECORDING_MIN)
    )
    return result
