"""
tests/test_normals.py
=====================

Fast, dependency-free checks for the *normals* package:

* :pyfunc:`mms_boundary_analysis.normals.mva_normal`
* :pyfunc:`mms_boundary_analysis.normals.angle_between`
* :pyfunc:`mms_boundary_analysis.normals.timing_normal`
* :pyfunc:`mms_boundary_analysis.normals.bootstrap_angles`
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mms_boundary_analysis.normals import (
    mva_normal,
    angle_between,
    timing_normal,
    bootstrap_angles,
)


# ---------------------------------------------------------------------
# mva_normal
# ---------------------------------------------------------------------
def test_mva_normal_rotating_xy() -> None:
    """
    A field that rotates *only* in the X–Y plane has its minimum‐variance
    direction along ±Z.
    """
    t = np.linspace(0.0, 2 * np.pi, 500)
    Bx = np.cos(t)
    By = np.sin(t)
    Bz = 0.02 * np.ones_like(t)        # small bias

    n_hat = mva_normal(np.c_[Bx, By, Bz])
    # sign is arbitrary – compare absolute dot with +Z
    assert abs(n_hat[2]) > 0.95, "normal should be close to ±Z"


# ---------------------------------------------------------------------
# angle_between
# ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "v1, v2, expected",
    [
        ([1, 0, 0], [1, 0, 0], 0.0),
        ([1, 0, 0], [0, 1, 0], 90.0),
        ([0, 0, 1], [0, 0, -1], 180.0),
    ],
)
def test_angle_between_basic(v1, v2, expected) -> None:
    angle = angle_between(np.array(v1), np.array(v2))
    assert math.isclose(angle, expected, abs_tol=1e-6)


# ---------------------------------------------------------------------
# timing_normal
# ---------------------------------------------------------------------
def test_timing_normal_known_plane() -> None:
    """
    Plane normal along +X; four spacecraft displaced arbitrarily in YZ.
    Crossing times are consistent with Vn = 10 km/s.
    """
    Vn_true = 10.0               # km/s
    n_true = np.array([1.0, 0.0, 0.0])

    # arbitrary Y–Z positions (km) – X chosen so t = x / Vn
    rs = np.array(
        [
            [100.0,  10.0,  0.0],
            [200.0, -12.0,  5.0],
            [150.0,   2.0, -6.0],
            [250.0,   0.0,  1.0],
        ]
    )
    ts = rs[:, 0] / Vn_true      # seconds

    n_hat, Vn_est, rms = timing_normal(rs, ts, return_speed=True)

    assert angle_between(n_hat, n_true) < 1.0  # deg
    assert abs(Vn_est - Vn_true) < 1e-2        # km/s
    assert rms < 1e-6                          # ideal synthetic data


# ---------------------------------------------------------------------
# bootstrap_angles
# ---------------------------------------------------------------------
def test_bootstrap_angles_low_scatter() -> None:
    """
    For a clean planar rotation the bootstrap 1-σ scatter should be small.
    """
    t = np.linspace(0.0, 2 * np.pi, 400)
    B = np.c_[np.sin(t), np.cos(t), 0.05 * np.ones_like(t)]
    sigma_deg = bootstrap_angles(B, n_iter=300, seed=42)
    assert sigma_deg < 5.0, "bootstrap scatter should be <5° for ideal data"
