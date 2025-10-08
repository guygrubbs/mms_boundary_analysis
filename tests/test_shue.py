from __future__ import annotations

import numpy as np
import pytest

from mms_boundary_analysis.distance import shue_radius
from mms_boundary_analysis.distance.shue import shue_normal
from mms_boundary_analysis.distance.projection import project_along_normal
from mms_boundary_analysis import config


def _shue_level_function(r: np.ndarray, P_dyn: float, Bz_nT: float) -> float:
    r = np.asarray(r, dtype=float)
    r_mag = np.linalg.norm(r)
    if r_mag == 0.0:
        return 0.0
    cos_theta = np.clip(r[0] / r_mag, -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(cos_theta))
    return r_mag - shue_radius(theta_deg, P_dyn, Bz_nT)


def test_shue_normal_matches_numerical_gradient():
    r0 = np.array([12.0, 3.0, -1.5]) * 6371.0
    P_dyn = 2.5
    Bz_nT = -1.2

    n_analytic = shue_normal(r0, P_dyn, Bz_nT)
    assert pytest.approx(1.0, rel=1e-6) == np.linalg.norm(n_analytic)

    eps = 1.0  # km
    grad = np.zeros(3)
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        grad[i] = (
            _shue_level_function(r0 + step, P_dyn, Bz_nT)
            - _shue_level_function(r0 - step, P_dyn, Bz_nT)
        ) / (2 * eps)

    n_numeric = grad / np.linalg.norm(grad)
    # Align orientation
    if np.dot(n_numeric, r0) < 0:
        n_numeric = -n_numeric

    cos_angle = np.clip(np.dot(n_numeric, n_analytic), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    assert angle < 1.0


def test_shue_normal_subsolar_fallback():
    r0 = np.array([10.0, 0.0, 0.0]) * 6371.0
    n = shue_normal(r0, 2.0, 0.0)
    assert np.allclose(n, np.array([1.0, 0.0, 0.0]))


def test_project_along_normal_converges_without_fallback():
    """Ensure the normal-projection root finder hits the surface."""

    Pd = 2.5
    Bz = -1.0

    theta_deg = 25.0
    phi_deg = 40.0
    r_surface = shue_radius(theta_deg, Pd, Bz)
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    r_vec = np.array([
        r_surface * np.cos(theta),
        r_surface * np.sin(theta) * np.cos(phi),
        r_surface * np.sin(theta) * np.sin(phi),
    ])

    n_hat = shue_normal(r_vec, Pd, Bz)
    displacement = 0.5 * config.RE_KM
    r0 = r_vec + displacement * n_hat  # move outward along the normal

    dN = float(project_along_normal(r0, n_hat, Pd, Bz, tol=1e-6, max_iter=100))
    assert np.isfinite(dN)

    r_on_surface = r0 + dN * n_hat
    theta_back = np.degrees(np.arccos(np.clip(r_on_surface[0] / np.linalg.norm(r_on_surface), -1.0, 1.0)))
    residual = np.linalg.norm(r_on_surface) - shue_radius(theta_back, Pd, Bz)

    assert abs(residual) < 1e-3
    assert pytest.approx(-displacement, rel=0.05, abs=2.0) == dN
