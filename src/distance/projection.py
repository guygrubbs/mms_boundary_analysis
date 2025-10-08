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


# ---------------------------------------------------------------------
# root-finder (scalar)  –  secant method with safety bisection
# ---------------------------------------------------------------------
def _root_secant(
    f,
    a: float,
    b: float,
    args: Tuple,
    tol: float,
    max_iter: int,
) -> float | None:
    """Bracketed secant solver with bisection safeguard."""

    fa = f(a, *args)
    fb = f(b, *args)

    if np.isnan(fa) or np.isnan(fb):
        return None
    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b
    if fa * fb > 0:  # no sign change → not bracketed
        return None

    for _ in range(max_iter):
        # Ensure |fa| >= |fb| so that b is our best approximation
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        denom = fb - fa
        if abs(denom) < 1e-12:
            c = 0.5 * (a + b)
        else:
            c = b - fb * (b - a) / denom  # secant step
            # keep the iterate inside the bracket
            if not (min(a, b) <= c <= max(a, b)):
                c = 0.5 * (a + b)

        fc = f(c, *args)
        if np.isnan(fc):
            return None
        if abs(fc) < tol:
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

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
    from .shue import _theta_from_r, shue_radius

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

        f0 = f_scalar(0.0, r0, n_hat, Pd, Bz)
        if np.isnan(f0):
            out[idx] = np.nan
            continue
        if abs(f0) < tol:
            out[idx] = 0.0
            continue

        bracket = None
        step = 0.25 * config.RE_KM
        max_span = 80 * config.RE_KM
        span = step

        while span <= max_span:
            s_pos = span
            f_pos = f_scalar(s_pos, r0, n_hat, Pd, Bz)
            if not np.isnan(f_pos) and f0 * f_pos <= 0:
                bracket = (0.0, s_pos)
                break

            s_neg = -span
            f_neg = f_scalar(s_neg, r0, n_hat, Pd, Bz)
            if not np.isnan(f_neg) and f0 * f_neg <= 0:
                bracket = (s_neg, 0.0)
                break

            span *= 2.0

        if bracket is not None:
            s = _root_secant(
                f_scalar,
                bracket[0],
                bracket[1],
                args=(r0, n_hat, Pd, Bz),
                tol=tol,
                max_iter=max_iter,
            )
        else:
            s = None

        if s is None:   # fallback radial
            theta = _theta_from_r(r0)
            dN = np.linalg.norm(r0) - shue_radius(theta, Pd, Bz)
            warnings.warn(
                "project_along_normal: bracket/solver failed; "
                "using radial difference",
                stacklevel=2,
            )
            out[idx] = dN
        else:
            out[idx] = s

    return out


__all__ = ["project_along_normal"]

