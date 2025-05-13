"""
mms_boundary_analysis.distance.projection
=========================================

Intersection *along a given normal* with the Shue-98 magnetopause
surface.

For each spacecraft position **r₀** (km) and unit normal **n̂** we solve
for the scalar *s* in

    **r(s)** = **r₀** + s **n̂**

such that the point **r(s)** lies **on** the Shue (1998) surface

    |**r(s)**| = r_model(θ, P_dyn, Bz).

The signed distance *ΔN = s* (km) is **positive** when moving *along*
*n̂* takes the spacecraft outward (larger |r|) and **negative** when it
moves inward.

If the root cannot be bracketed the routine falls back to the simpler
radial method (|r| − r_model) and issues a warning.

Public function
---------------

``project_along_normal(r_sc, n_hat, P_dyn, Bz_nT,
                       tol=1e-3, max_iter=50)``  → ΔN array (km)
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from .. import config
from .shue import _theta_from_r, shue_radius


# ---------------------------------------------------------------------
# root-finder (scalar)  –  secant method with safety bisection
# ---------------------------------------------------------------------
def _root_secant(
    f,
    a: float,
    b: float,
    args: Tuple,
    tol: float,
    max_iter: int
) -> float | None:
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:          # not bracketed
        return None

    for _ in range(max_iter):
        if abs(fb - fa) < 1e-12:   # avoid division by zero
            break
        c = b - fb * (b - a) / (fb - fa)   # secant step
        fc = f(c, *args)

        if abs(fc) < tol:
            return c

        # maintain bracket
        a, fa = (b, fb) if fa * fc < 0 else (a, fa)
        b, fb = c, fc

    return None


# ---------------------------------------------------------------------
# signed distance along n̂
# ---------------------------------------------------------------------
def project_along_normal(
    r_sc: np.ndarray,
    n_hat: np.ndarray,
    P_dyn: float | np.ndarray,
    Bz_nT: float | np.ndarray,
    /,
    *,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Distance (km) to Shue surface along the supplied normal vector.

    Parameters
    ----------
    r_sc : ndarray, shape (..., 3)
        Spacecraft position(s) in kilometres (GSE or GSM).
    n_hat : ndarray, shape (3,)
        Unit normal vector (boundary‐normal direction).
    P_dyn, Bz_nT : float or ndarray broadcastable to r_sc[...,0]
        Solar-wind dynamic pressure (nPa) and IMF Bz (nT).
    tol : float, default 1e-3
        Convergence tolerance on |f(s)| (km).
    max_iter : int, default 50
        Maximum iterations in secant solver.

    Returns
    -------
    ndarray
        Signed distances ΔN (km).  Positive ⇒ outward along *n̂*.
    """
    r_sc = np.asarray(r_sc, dtype=float)
    if r_sc.shape[-1] != 3:
        raise ValueError("r_sc must be (..., 3)")

    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    # broadcast inputs
    P_dyn  = np.broadcast_to(P_dyn,  r_sc.shape[:-1])
    Bz_nT  = np.broadcast_to(Bz_nT,  r_sc.shape[:-1])
    out    = np.empty(P_dyn.shape, dtype=float)

    # scalar helper for root-finding
    def f_scalar(s, r0, n, Pd, Bz):
        r = r0 + s * n
        theta = _theta_from_r(r)
        return np.linalg.norm(r) - shue_radius(theta, Pd, Bz)

    # iterate over flattened index space
    flat_iter = np.ndindex(P_dyn.shape)
    for idx in flat_iter:
        r0   = r_sc[idx]        # (3,)
        Pd   = float(P_dyn[idx])
        Bz   = float(Bz_nT[idx])

        # simple bracketing: try ±5 R_E from current position
        RE = 6371.0
        s_min, s_max = -5 * RE, 5 * RE

        s = _root_secant(f_scalar, s_min, s_max,
                         args=(r0, n_hat, Pd, Bz),
                         tol=tol, max_iter=max_iter)

        if s is None:   # fallback radial
            theta = _theta_from_r(r0)
            dN = np.linalg.norm(r0) - shue_radius(theta, Pd, Bz)
            warnings.warn("project_along_normal: secant failed; "
                          "using radial difference")
            out[idx] = dN
        else:
            out[idx] = s

    return out


__all__ = ["project_along_normal"]
