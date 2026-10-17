import numpy as np
import pytest

from src.features import extract_ctg_features


def test_flat_signal_has_expected_baseline_and_no_events():
    fhr = np.full(240, 140.0)
    uc = np.zeros(240)

    features = extract_ctg_features(fhr, uc, fs=1.0)

    assert features["baseline_fhr_bpm"] == pytest.approx(140.0)
    assert features["fhr_std_bpm"] == pytest.approx(0.0)
    assert features["n_accelerations"] == 0
    assert features["n_decelerations"] == 0


def test_detects_acceleration_and_deceleration():
    fhr = np.full(240, 140.0)
    fhr[30:50] = 160.0
    fhr[100:125] = 115.0
    uc = np.zeros(240)

    features = extract_ctg_features(fhr, uc, fs=1.0)

    assert features["n_accelerations"] == 1
    assert features["n_decelerations"] == 1
    assert features["deceleration_mean_depth_bpm"] == pytest.approx(25.0)


def test_detects_contraction_and_late_deceleration():
    fhr = np.full(360, 140.0)
    fhr[180:205] = 115.0
    uc = np.zeros(360)
    uc[120] = 10.0

    features = extract_ctg_features(fhr, uc, fs=1.0)

    assert features["n_contractions"] == 1
    assert features["n_late_decelerations"] == 1
