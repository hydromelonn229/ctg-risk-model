import numpy as np
import pytest

from src.preprocessing import (
    clean_fhr,
    interpolate_short_gaps,
    select_analysis_window,
)


def test_marks_implausible_values_and_interpolates_short_gap():
    fhr = np.array([140.0, 0.0, 150.0, 145.0, 230.0])

    cleaned, quality = clean_fhr(fhr, fs=1.0, max_gap_s=2.0)

    assert cleaned[1] == pytest.approx(145.0)
    assert np.isnan(cleaned[4])
    assert quality["interpolated_samples"] == 1
    assert quality["pct_signal_loss_before_interpolation"] == pytest.approx(40.0)


def test_leaves_long_gaps_missing():
    fhr = np.array([140.0, np.nan, np.nan, np.nan, 150.0])

    cleaned = interpolate_short_gaps(fhr, fs=1.0, max_gap_s=2.0)

    assert np.isnan(cleaned[1:4]).all()


def test_does_not_interpolate_edge_gaps():
    fhr = np.array([np.nan, 140.0, 145.0, np.nan])

    cleaned = interpolate_short_gaps(fhr, fs=1.0, max_gap_s=2.0)

    assert np.isnan(cleaned[[0, 3]]).all()


def test_selects_final_window():
    signal = np.arange(10)

    window = select_analysis_window(signal, fs=1.0, window_min=0.1)

    np.testing.assert_array_equal(window, np.array([4, 5, 6, 7, 8, 9]))


def test_rejects_signal_shorter_than_window():
    with pytest.raises(ValueError, match="needs"):
        select_analysis_window(np.arange(5), fs=1.0, window_min=0.1)
