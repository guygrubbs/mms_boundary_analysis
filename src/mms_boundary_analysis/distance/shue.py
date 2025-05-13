"""
mms_boundary_analysis.distance.shue
===================================

Global *model* distance to the magnetopause using the empirical
**Shue et al. (1998)** surface

.. math::

    r(\\theta) = r_0 \\left[ \\frac{2}{1 + \\cos\\theta} \\right]^{\\alpha}

where :math:`\\theta` is the angle from the Sun–Earth line and

.. math::

    r_0 = a_0 P_{\\mathrm{dyn}}^{a_1} + a_2 B_z, \\qquad
    \\alpha = b_0 + b_1 B_z.

The coefficients :math:`a_0, a_1, a_2, b_0, b_1` are read from
:pydata:`mms_boundary_analysis.config.SHUE_COEFFS`.

Public helpers
--------------

``shue_radius(theta_deg, P_dyn, Bz_nT)``
    Forward to :pyfunc:`mms_boundary_analysis.config.shue_radius`
    for convenience.

``delta_n_shue(r_sc, n_hat, P_dyn, Bz_nT, /, radial=False)``
    Signed distance (km) from the spacecraft to the model magnetopause.

    * If *radial=True* (default) the distance is simply
      :math:`|\\mathbf{r}| - r(\\theta_{sc})` where
      :math:`\\theta_{sc}` is computed from the spacecraft GSE vector.
    * If *radial=False* the routine projects **along the supplied
      normal** by calling :pyfunc:`mms_boundary_analysis.distance.
      projection.project_along_normal` (falls back to radial method if
      that helper is unavailable).

Both functions are vectorised: *r_sc* can be a single 3–vector or an
``(N, 3)`` array.
"""

from __future__ import annotations

from typing import Tuple, overload

import numpy as np

from .. import config

# optional import – keep soft dependency
try:
    from .projection import project_along_normal   # noqa: F401
except ModuleNotFoundError:  # projection.py not yet implemented
    project_along_normal = None     # type: ignore


# ----------------------------------------------------------------------
# convenience passthrough
# ----------------------------------------------------------------------
shue_radius = config.shue_radius


# ----------------------------------------------------------------------
# compute theta (deg) from GSE position
# ----------------------------------------------------------------------
def _theta_from_r(r: np.ndarray) -> np.ndarray:
    """
    Return polar angle θ (deg) from +X_GSE axis.

    Parameters
    ----------
    r : ndarray, shape (..., 3)

    Returns
    -------
    ndarray, shape (...,)
        θ in degrees, 0° = subsolar point, 180° = anti‐solar.
    """
    r = np.asarray(r, dtype=float)
    r_norm = np.linalg.norm(r, axis=-1)
    # Avoid divide-by-zero
    cos_theta = np.divide(
        r[..., 0], r_norm, out=np.zeros_like(r_norm), where=r_norm != 0.0
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# ----------------------------------------------------------------------
# signed distance to Shue surface
# ----------------------------------------------------------------------
@overload
def delta_n_shue(
    r_sc: np.ndarray,
    n_hat: np.ndarray,
    P_dyn: float,
    Bz_nT: float,
    /,
    *,
    radial: bool = ...
) -> np.ndarray: ...
@overload
def delta_n_shue(
    r_sc: np.ndarray,
    n_hat: np.ndarray,
    P_dyn: np.ndarray,
    Bz_nT: np.ndarray,
    /,
    *,
    radial: bool = ...
) -> np.ndarray: ...


def delta_n_shue(
    r_sc: np.ndarray,
    n_hat: np.ndarray,
    P_dyn: float | np.ndarray,
    Bz_nT: float | np.ndarray,
    /,
    *,
    radial: bool = True
) -> np.ndarray:
    """
    Signed gap between spacecraft and Shue‐98 magnetopause (km).

    Parameters
    ----------
    r_sc : ndarray, shape (..., 3)
        Spacecraft GSE (or GSM) position in **kilometres**.
    n_hat : ndarray, shape (3,)
        Event normal (unit).  Used only if *radial=False*.
    P_dyn, Bz_nT : float or ndarray broadcastable to r_sc[...,0]
        Solar-wind dynamic pressure (nPa) and IMF Bz (nT).
    radial : bool, default ``True``
        * True – use simple radial difference |r| − r(θ).
        * False – project along *n_hat* to intersect the model surface.
          Requires :pyfunc:`projection.project_along_normal`; if that
          module is missing falls back to the radial method.

    Returns
    -------
    ndarray
        Signed distance(s) in kilometres (same leading shape as *r_sc*).
        Positive ⇒ spacecraft outside the model MP.
    """
    r_sc = np.asarray(r_sc, dtype=float)
    if r_sc.shape[-1] != 3:
        raise ValueError("r_sc must have last-dimension length 3")

    if radial or project_along_normal is None:
        theta = _theta_from_r(r_sc)
        r_model = shue_radius(theta, P_dyn, Bz_nT)
        r_mag = np.linalg.norm(r_sc, axis=-1)
        return r_mag - r_model            # positive => outside
    else:
        # more accurate: intersection along supplied normal
        return project_along_normal(r_sc, n_hat, P_dyn, Bz_nT)


__all__ = ["shue_radius", "delta_n_shue"]
