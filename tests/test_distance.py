"""
tests/test_distance.py
======================

Unit-tests for the geometry helpers in *mms_boundary_analysis.distance*.

The tests deliberately stick to **pure NumPy** maths so they run quickly
and have zero external dependencies – ideal for CI pipelines.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mms_boundary_analysis.distance import (
    delta_n_local,
    shue_radius,
    delta_n_shue,
)

# Earth radius (km) used by Shue helper
RE = 6371.0


# ---------------------------------------------------------------------
# delta_n_local
# ---------------------------------------------------------------------
def test_delta_n_local_simple_line() -> None:
    """
    Straight line along +X with normal also +X ⇒ ΔN is just (x − x0).
    """
    r = np.array([[10.0, 0.0, 0.0],
                  [11.5, 0.0, 0.0],
                  [14.2, 0.0, 0.0]])  # km
    r0 = r[0]
    n  = np.array([1.0, 0.0, 0.0])

    dN = delta_n_local(r, r0, n)
    assert np.allclose(dN, [0.0, 1.5, 4.2]), "ΔN should equal Δx for this setup"


def test_delta_n_local_oblique() -> None:
    """
    Plane normal (1,1,0) – check signed distance for two symmetric points.
    """
    n = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
    r0 = np.array([0.0, 0.0, 0.0])

    r_plus  = np.array([[ 1.0,  1.0, 0.0]])   # +√2 along n
    r_minus = np.array([[-1.0, -1.0, 0.0]])   # −√2 along n
    assert math.isclose(delta_n_local(r_plus,  r0, n)[0],  math.sqrt(2),  rel_tol=1e-10)
    assert math.isclose(delta_n_local(r_minus, r0, n)[0], -math.sqrt(2),  rel_tol=1e-10)


# ---------------------------------------------------------------------
# shue_radius
# ---------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0, 30, 90, 150])
def test_shue_radius_monotonic(theta_deg: float) -> None:
    """
    For fixed Pdyn & Bz the Shue radius should monotonically *increase*
    with θ (sub-solar → flank).
    """
    r0   = shue_radius(0.0, 1.5,  0.0)        # sub-solar point
    r_th = shue_radius(theta_deg, 1.5, 0.0)
    assert r_th >= r0, "r(θ) must grow away from nose"


def test_shue_radius_bz_dependence() -> None:
    """
    Southward IMF (negative Bz) compresses the magnetopause – r0 smaller.
    """
    r_Bz0 = shue_radius(0.0, 1.5,  0.0)   # Bz =  0 nT
    r_Bzn = shue_radius(0.0, 1.5, -5.0)   # Bz = -5 nT
    assert r_Bzn < r_Bz0, "negative Bz should reduce sub-solar standoff"


# ---------------------------------------------------------------------
# delta_n_shue  (radial branch only)
# ---------------------------------------------------------------------
def test_delta_n_shue_radial_sign() -> None:
    """
    Spacecraft placed 1 R_E *outside* the sub-solar model surface should
    yield positive ΔN; inside should be negative.
    """
    Pd, Bz = 1.8, 0.0
    r_nose = shue_radius(0.0, Pd, Bz)

    r_out = np.array([r_nose + RE, 0.0, 0.0])   # 1 R_E outward
    r_in  = np.array([r_nose - RE, 0.0, 0.0])   # 1 R_E inward

    n_hat = np.array([1.0, 0.0, 0.0])

    dN_out = delta_n_shue(r_out, n_hat, Pd, Bz, radial=True)
    dN_in  = delta_n_shue(r_in,  n_hat, Pd, Bz, radial=True)

    assert dN_out >  0, "outside point should give +ΔN"
    assert dN_in  <  0, "inside  point should give -ΔN"
