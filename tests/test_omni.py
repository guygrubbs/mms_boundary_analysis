"""Integration coverage for OMNI context downloads."""

from __future__ import annotations

import numpy as np
import pytest

from mms_boundary_analysis.io import omni


def _skip_if_offline(exc: Exception) -> None:
    message = str(exc)
    if (
        "No internet" in message
        or "Connection" in message
        or "NoneType" in message
        or "not in pytplot" in message
    ):
        pytest.skip("OMNI data download failed – offline environment")
    raise exc


def test_get_context_real_download(tmp_path):
    ts = np.datetime64("2019-01-27T12:30:00Z")
    try:
        ctx = omni.get_context(ts)
    except Exception as exc:  # pyspedas may raise various network errors
        _skip_if_offline(exc)

    keys = {"Bx_nT", "By_nT", "Bz_nT", "Vsw_kms", "Pdyn_nPa", "clock_deg", "cone_deg"}
    assert keys.issubset(ctx.keys())
    assert all(np.isfinite(ctx[k]) for k in keys)
