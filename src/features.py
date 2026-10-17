"""Clinically named, deliberately simplified CTG feature extraction."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.data_loading import FS
from src.preprocessing import prepare_recording

ACCEL_THRESHOLD_BPM = 15.0
DECEL_THRESHOLD_BPM = 15.0
MIN_EVENT_DURATION_S = 15.0
CONTRACTION_MIN_DISTANCE_S = 60.0
LATE_LAG_MIN_S = 10.0
LATE_LAG_MAX_S = 90.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _event_summary(
    fhr: np.ndarray,
    baseline: float,
    direction: str,
    fs: float,
    threshold_bpm: float = DECEL_THRESHOLD_BPM,
) -> list[dict[str, float]]:
    if direction == "up":
        mask = fhr >= baseline + threshold_bpm
    elif direction == "down":
        mask = fhr <= baseline - threshold_bpm
    else:
        raise ValueError("direction must be 'up' or 'down'")

    events = []
    min_samples = int(round(MIN_EVENT_DURATION_S * fs))
    for start, end in _runs(np.nan_to_num(mask, nan=False)):
        if end - start < min_samples:
            continue
        segment = fhr[start:end]
        if np.isnan(segment).all():
            continue
        nadir_or_peak = np.nanmin(segment) if direction == "down" else np.nanmax(segment)
        events.append(
            {
                "start": float(start),
                "end": float(end),
                "duration_s": (end - start) / fs,
                "depth_bpm": max(0.0, baseline - nadir_or_peak),
                "peak_index": float(start + np.nanargmax(segment)),
                "nadir_index": float(start + np.nanargmin(segment)),
            }
        )
    return events


def _baseline(fhr: np.ndarray) -> float:
    valid = fhr[~np.isnan(fhr)]
    return float(np.nanmedian(valid)) if len(valid) else np.nan


def _variability_features(fhr: np.ndarray, fs: float) -> dict[str, float]:
    valid = fhr[~np.isnan(fhr)]
    if not len(valid):
        return {"fhr_std_bpm": np.nan, "short_term_variability_bpm": np.nan,
                "long_term_variability_bpm": np.nan}
    adjacent = fhr[1:] - fhr[:-1]
    adjacent = adjacent[~np.isnan(adjacent)]
    window_samples = max(1, int(round(60 * fs)))
    ltv = [
        np.nanmax(fhr[start:start + window_samples])
        - np.nanmin(fhr[start:start + window_samples])
        for start in range(0, len(fhr), window_samples)
        if np.isfinite(fhr[start:start + window_samples]).any()
    ]
    return {
        "fhr_std_bpm": float(np.nanstd(valid)),
        "short_term_variability_bpm": float(np.nanmean(np.abs(adjacent)))
        if len(adjacent) else np.nan,
        "long_term_variability_bpm": float(np.nanmean(ltv)) if ltv else np.nan,
    }


def _contraction_peaks(uc: np.ndarray, fs: float) -> np.ndarray:
    valid = uc[np.isfinite(uc)]
    if not len(valid):
        return np.array([], dtype=int)
    baseline = np.nanmedian(valid)
    amplitude = np.nanpercentile(valid, 95) - baseline
    prominence = max(float(amplitude * 0.25), float(np.nanstd(valid) * 0.5), 1e-6)
    peaks, _ = find_peaks(
        np.nan_to_num(uc, nan=baseline),
        distance=int(round(CONTRACTION_MIN_DISTANCE_S * fs)),
        prominence=prominence,
    )
    return peaks


def extract_ctg_features(
    fhr: np.ndarray,
    uc: np.ndarray,
    fs: float = FS,
) -> dict[str, float]:
    """Extract one row of interpretable features from a cleaned CTG window."""
    if len(fhr) != len(uc):
        raise ValueError("FHR and UC must have the same number of samples")

    baseline = _baseline(fhr)
    features = {"baseline_fhr_bpm": baseline}
    features.update(_variability_features(fhr, fs))

    accelerations = _event_summary(fhr, baseline, "up", fs, ACCEL_THRESHOLD_BPM)
    decelerations = _event_summary(fhr, baseline, "down", fs, DECEL_THRESHOLD_BPM)
    features.update(
        {
            "n_accelerations": float(len(accelerations)),
            "acceleration_duration_s": float(
                sum(event["duration_s"] for event in accelerations)
            ),
            "n_decelerations": float(len(decelerations)),
            "deceleration_mean_depth_bpm": float(
                np.mean([event["depth_bpm"] for event in decelerations])
            ) if decelerations else 0.0,
            "deceleration_mean_duration_s": float(
                np.mean([event["duration_s"] for event in decelerations])
            ) if decelerations else 0.0,
        }
    )

    contraction_peaks = _contraction_peaks(uc, fs)
    late_count = 0
    for contraction_peak in contraction_peaks:
        lagged = [
            event for event in decelerations
            if contraction_peak + LATE_LAG_MIN_S * fs
            <= event["nadir_index"]
            <= contraction_peak + LATE_LAG_MAX_S * fs
        ]
        late_count += bool(lagged)
    features["n_contractions"] = float(len(contraction_peaks))
    features["contraction_rate_per_10min"] = float(
        len(contraction_peaks) / 3.0
    )
    features["n_late_decelerations"] = float(late_count)
    features["pct_signal_missing"] = float(np.isnan(fhr).mean() * 100)
    return features


def build_feature_table(
    raw_dir: Path,
    metadata: pd.DataFrame,
    fs: float = FS,
) -> pd.DataFrame:
    """Extract features for all records and join them to metadata."""
    rows = []
    for record_name in metadata["record"]:
        fhr, uc, quality = prepare_recording(record_name, raw_dir, fs=fs)
        features = extract_ctg_features(fhr, uc, fs=fs)
        features["record"] = record_name
        rows.append(features)
    feature_df = pd.DataFrame(rows)
    return metadata.merge(feature_df, on="record", how="inner", validate="one_to_one")
