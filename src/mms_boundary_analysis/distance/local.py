"""
mms_boundary_analysis.distance.local
====================================

*Local* distance to the magnetopause (or any planar boundary).

For a given boundary‐normal vector **n̂** and a reference point **r₀**
(usually the spacecraft position at the **current‐sheet crossing
time**), the signed distance of the spacecraft at any later instant is

    ΔN(t) = ( **r(t)** − **r₀** ) · **n̂**                       (1)

Positive ΔN means the spacecraft moved *along* the chosen normal
direction, negative means the opposite.  Units are the same as **r**
(km in our pipeline).

This file offers thin wrappers that operate on NumPy vectors, so the
function can be applied to whole time series with one call.

Example
-------
```python
import numpy as np
from mms_boundary_analysis.distance import delta_n_local

# 3-point trajectory (km)
r  = np.array([[10, 0, 0],
               [11, 0, 0],
               [12, 0, 0]])
r0 = r[1]                           # reference: second sample
n  = np.array([1, 0, 0])            # outward along +X

dN = delta_n_local(r, r0, n)
print(dN)            # → [-1, 0, +1]  km
````
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------

# core helper

# ---------------------------------------------------------------------

def delta_n_local(
    r: np.ndarray,
    r0: np.ndarray,
    n_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute signed distance ΔN(t) = (r − r0)·n̂ for an entire trajectory.

    ```
    Parameters
    ----------
    r : ndarray, shape (N, 3)
        Spacecraft position vector(s) in kilometres.
    r0 : ndarray, shape (3,)
        Reference position (km) – normally the point of boundary
        intersection (ΔN = 0 there).
    n_hat : ndarray, shape (3,)
        Unit normal vector for the boundary.

    Returns
    -------
    ndarray, shape (N,)
        Signed distance time series in kilometres.
    """
    r = np.asarray(r, dtype=float)
    r0 = np.asarray(r0, dtype=float).reshape(1, 3)
    n_hat = np.asarray(n_hat, dtype=float) / np.linalg.norm(n_hat)

    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("r must have shape (N, 3)")
    if n_hat.shape != (3,):
        raise ValueError("n_hat must be a 3-vector")

    # Broadcast r0 over rows of r
    return np.einsum("ij,j->i", r - r0, n_hat)


__all__ = ["delta_n_local"]

"""
**Key points**

* Pure NumPy implementation for speed (`einsum` avoids explicit loops).
* Validates shapes and normalises the input normal vector.
* Works on single points (`N = 1`) or long trajectories (`N ≫ 1`).
"""
