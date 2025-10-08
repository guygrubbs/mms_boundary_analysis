"""Regression tests for the visualisation helpers."""

from __future__ import annotations

import importlib

import numpy as np
import matplotlib
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from mms_boundary_analysis.visual.timeseries import plot_timeseries
from mms_boundary_analysis.visual.imf_scatter import plot_imf_scatter
from mms_boundary_analysis.visual.spectrograms import (
    plot_fpi_spectrogram,
    plot_fgm_components,
)

TRANGE = ["2019-01-27/12:00:00", "2019-01-27/12:10:00"]


def _skip_if_offline(exc: Exception) -> None:
    message = str(exc)
    if (
        "No internet" in message
        or "Connection" in message
        or "unavailable after download attempt" in message
        or "not in pytplot" in message
    ):
        pytest.skip("MMS plotting data unavailable – offline environment")
    raise exc


def test_timeseries_module_imports() -> None:
    """Ensure the timeseries helper can be imported without config errors."""

    module = importlib.import_module("mms_boundary_analysis.visual.timeseries")
    assert hasattr(module, "plot_timeseries")


def test_plot_timeseries_handles_nan_and_sorting() -> None:
    """Plotting should tolerate NaNs and unsorted time arrays."""

    times = np.array([
        np.datetime64("2019-01-27T12:00:30Z"),
        np.datetime64("2019-01-27T12:00:00Z"),
        np.datetime64("2019-01-27T12:00:15Z"),
    ])
    local = np.array([np.nan, 0.0, 5.0])
    model = np.array([np.nan, np.nan, np.nan])
    posix = ((times - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")).astype(float)

    dseries = {
        "mms1": {"time": times, "local": local, "model": model},
        "mms2": {"time": np.array([0.0, 1.0]), "local": np.array([np.nan, np.nan])},
    }
    events = {"mms1": [{"time": float(posix[1]), "category": "MP full"}]}

    fig, ax = plot_timeseries(dseries, events)

    _handles, labels = ax.get_legend_handles_labels()
    assert "MP full" in labels
    assert "MMS2  ΔN local" not in labels

    # Only the MMS1 local curve should be rendered; model is fully NaN.
    lines = ax.get_lines()
    assert any("MMS1" in line.get_label() for line in lines)

    plt.close(fig)


def test_plot_imf_scatter_handles_empty_events() -> None:
    """Scatter plot should not crash when no events are present."""

    events = {"mms1": []}
    fig, ax = plot_imf_scatter(events, title="IMF overview")

    assert any("No events" in txt.get_text() for txt in ax.texts)
    assert len(ax.collections) == 0

    plt.close(fig)


def test_plot_imf_scatter_filters_nan_metrics() -> None:
    """Only events with numeric metrics should appear in the scatter plot."""

    events = {
        "mms1": [
            {"clock_deg": 30.0, "category": "MP full"},
            {"clock_deg": np.nan, "category": "unknown"},
        ]
    }

    fig, ax = plot_imf_scatter(events)
    # Only one point should be rendered
    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    assert offsets.shape[0] == 1

    plt.close(fig)


def test_plot_fpi_spectrogram_real_data() -> None:
    try:
        fig, axes = plot_fpi_spectrogram("1", TRANGE)
    except Exception as exc:
        _skip_if_offline(exc)

    for ax in axes:
        assert ax.has_data()
    plt.close(fig)


def test_plot_fgm_components_real_data() -> None:
    try:
        fig, axes = plot_fgm_components("1", TRANGE)
    except Exception as exc:
        _skip_if_offline(exc)

    for ax in axes:
        lines = ax.get_lines()
        assert lines, "Expected magnetic-field traces"
    plt.close(fig)
