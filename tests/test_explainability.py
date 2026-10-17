import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    _apply_calibrator,
    _fit_calibrator,
    _calibration_metrics,
)


def test_platt_calibration_returns_probabilities():
    raw = np.linspace(0.05, 0.95, 20)
    y = np.array([0] * 10 + [1] * 10)
    calibrator = _fit_calibrator("platt", raw, y)
    calibrated = _apply_calibrator(calibrator, "platt", raw)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    assert calibrated[-1] > calibrated[0]


def test_isotonic_calibration_returns_probabilities():
    raw = np.linspace(0.05, 0.95, 20)
    y = np.array([0] * 10 + [1] * 10)
    calibrator = _fit_calibrator("isotonic", raw, y)
    calibrated = _apply_calibrator(calibrator, "isotonic", raw)
    assert np.all((calibrated >= 0) & (calibrated <= 1))


def test_calibration_metrics_include_prevalence_and_brier():
    metrics = _calibration_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert metrics["observed_prevalence"] == 0.5
    assert metrics["brier"] == pytest.approx(0.04)
