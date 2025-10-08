"""Integration tests that rely on real MMS downloads when available."""

from __future__ import annotations

import numpy as np
import pytest

from mms_boundary_analysis.io import loader
from mms_boundary_analysis import config

TRANGE = ["2019-01-27/12:00:00", "2019-01-27/12:10:00"]
PROBES = ["1"]


def _skip_if_offline(exc: Exception) -> None:
    message = str(exc)
    if (
        "No internet" in message
        or "essential variables" in message
        or "No matching CDF" in message
    ):
        pytest.skip("MMS CDFs unavailable – likely offline test environment")
    raise exc


def test_load_mms_data_real_values():
    try:
        data = loader.load_mms_data(trange=TRANGE, probes=PROBES)
    except RuntimeError as exc:  # download failure surfaces as runtime error
        _skip_if_offline(exc)

    assert set(data.keys()) == {"mms1"}
    entry = data["mms1"]

    for key in ("time_pos", "pos", "time_vi", "Vi"):
        assert key in entry
        assert entry[key].size > 0

    # Ensure time arrays respect requested window
    start_sec = ((np.datetime64("2019-01-27T12:00:00Z") - config.EPOCH64) / np.timedelta64(1, "s")).astype(float)
    stop_sec = ((np.datetime64("2019-01-27T12:10:00Z") - config.EPOCH64) / np.timedelta64(1, "s")).astype(float)
    time_pos = entry["time_pos"]
    time_vi = entry["time_vi"]
    assert time_pos[0] >= start_sec - 1
    assert time_pos[-1] <= stop_sec + 1
    assert time_vi[0] >= start_sec - 1
    assert time_vi[-1] <= stop_sec + 1

    # MEC positions should be in kilometres (values larger than 1e3)
    pos_mag = np.linalg.norm(entry["pos"], axis=1)
    assert np.nanmedian(pos_mag) > 1e3
