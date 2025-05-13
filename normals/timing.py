"""
mms_boundary_analysis.normals.timing
====================================

Four–spacecraft **timing analysis** for planar boundaries
(see e.g. Russell et al., 1983; Harvey, 1998; Dunlop & Wood, 2015).

Let the boundary be a plane that at time *t\_0* passes spacecraft *i*
at position **rᵢ**.  If the plane moves rigidly with constant speed
*Vₙ* along its normal **n̂**, then for any two spacecraft *i,j*:

    (**rⱼ** − **rᵢ**) · **n̂**  =  Vₙ (tⱼ − tᵢ)             (1)

With ≥3 independent spacecraft we can solve for the normal and speed in
a least–squares sense.

This module provides a single helper:

``timing_normal(rs, ts, /, *, return_speed=False)``

where *rs* is an (N, 3) array of GSE/GSM positions (km) and
*ts* is a length-N array of times (any units convertible to seconds).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------
# core solver
# ---------------------------------------------------------------------
def timing_normal(
    rs: np.ndarray,
    ts: np.ndarray,
    *,
    return_speed: bool = False
) -> "np.ndarray | Tuple[np.ndarray, float, float]":
    """
    Least–squares boundary normal from multi-spacecraft timing.

    Parameters
    ----------
    rs : ndarray, shape (N_sc, 3)
        Spacecraft position vectors (km).
    ts : ndarray, shape (N_sc,)
        Crossing times – either ``numpy.datetime64`` or float seconds.
    return_speed : bool, default ``False``
        If *True* also return Vn (km s⁻¹) and RMS misfit (km).

    Returns
    -------
    n_hat : ndarray, shape (3,)
        Unit normal pointing in the direction of motion (sign
        convention: chosen so *Vn* is **positive**).
    Vn : float  (only if *return_speed*)
        Boundary speed along *n̂* (km s⁻¹).
    rms : float  (only if *return_speed*)
        Root-mean-square residual of Eq. (1) (km).

    Notes
    -----
    * If only three spacecraft are supplied the system is *exact*;
      with four it is over-determined and solved in a least-squares
      sense.
    * The solution is unique up to a global sign.  We choose the sign
      such that *Vn > 0* (outward motion).
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 2 or rs.shape[1] != 3 or rs.shape[0] < 3:
        raise ValueError("rs must be (N≥3, 3) array")

    # convert times to float seconds
    if np.issubdtype(np.asarray(ts).dtype, "datetime64"):
        ts = ts.astype("datetime64[ns]").astype("int64") * 1e-9  # ns → s
    ts = np.asarray(ts, dtype=float)
    if ts.shape != (rs.shape[0],):
        raise ValueError("ts must have same length as rs")

    # Construct pair-wise dr and dt
    rows = []
    rhs  = []
    n_sc = rs.shape[0]
    for i in range(n_sc - 1):
        for j in range(i + 1, n_sc):
            rows.append(rs[j] - rs[i])        # Δr  (km)
            rhs.append(ts[j] - ts[i])         # Δt  (s)

    A = np.vstack(rows)           # shape (N_pairs, 3)
    b = np.asarray(rhs)           # shape (N_pairs,)

    # Solve  A · (n / Vn) ≈ b   → x = n / Vn
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Speed and unit normal
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        raise RuntimeError("Degenerate timing solution (norm=0)")

    Vn  = 1.0 / norm_x
    n_hat = x * Vn                # = x / ||x||  (unit)

    # Ensure positive speed (flip if necessary)
    if Vn < 0:
        n_hat = -n_hat
        Vn = -Vn

    if return_speed:
        residuals = A @ n_hat - Vn * b
        rms = float(np.sqrt(np.mean(residuals**2)))
        return n_hat, float(Vn), rms
    return n_hat


__all__ = ["timing_normal"]
