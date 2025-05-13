"""
mms_boundary_analysis.normals.bootstrap
=======================================

Quick–and–dirty **bootstrap uncertainty** for boundary normals derived
with single–spacecraft MVA.

Given a magnetic‐field array *B* (shape *(N, 3)*), the routine:

1.  Draws *N* samples **with replacement** from *B* (bootstrap resample).
2.  Computes an MVA normal for the resample.
3.  Repeats steps 1–2 *n_iter* times (default 200).
4.  Returns the **1-σ angular spread** (in degrees) of the bootstrap
    normals about their mean direction.

Sign ambiguities (±n̂) are resolved by forcing every bootstrap normal to
point roughly the same way as the first one.

Example
-------
```python
from mms_boundary_analysis.normals import mva_normal, bootstrap_angles
import numpy as np

# Synthetic boundary
t  = np.linspace(0, 2*np.pi, 400)
B  = np.c_[np.sin(t), np.cos(t), 0.1*np.ones_like(t)]

n_hat = mva_normal(B)
sigma_deg = bootstrap_angles(B, n_iter=500)

print("Normal:", n_hat.round(3))
print("1-σ Δθ:", sigma_deg, "deg")
````

"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .mva import mva_normal, angle_between

# ---------------------------------------------------------------------

# public helper

# ---------------------------------------------------------------------

def bootstrap_angles(B: np.ndarray, *, n_iter: int = 200, seed: int | None = None, return_all: bool = False) -> float | Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate 1-σ angular scatter of the MVA normal via bootstrap.

    Parameters
    ----------
    B : ndarray, shape (N, 3)
        Magnetic-field samples used for MVA.
    n_iter : int, default 200
        Number of bootstrap resamples.
    seed : int or None
        Seed for NumPy RNG (for reproducibility).
    return_all : bool, default ``False``
        If *True*, also return the full array of angles *and* the
        bootstrap normals.

    Returns
    -------
    sigma_deg : float
        Standard deviation (deg) of angle between each bootstrap normal
        and their mean normal.
    angles_deg : ndarray, shape (n_iter,)   (only if *return_all*)
        Individual angles (deg) to mean normal.
    normals : ndarray, shape (n_iter, 3)    (only if *return_all*)
        Bootstrap normal vectors (unit length).
    """
    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3 or B.shape[0] < 4:
        raise ValueError("B must be (N>=4, 3)")

    rng = np.random.default_rng(seed)
    N = B.shape[0]

    normals = np.empty((n_iter, 3))
    ref = None

    for k in range(n_iter):
        idx = rng.integers(0, N, size=N)      # resample with replacement
        n = mva_normal(B[idx])

        if ref is None:
            ref = n
        # resolve ± ambiguity: flip if pointing opposite first normal
        if np.dot(n, ref) < 0:
            n = -n

        normals[k] = n

    # Mean direction
    mean_vec = normals.mean(axis=0)
    mean_vec /= np.linalg.norm(mean_vec)

    # Angle of each bootstrap normal to mean
    angles = np.array([angle_between(mean_vec, n) for n in normals])
    sigma = float(np.std(angles))

    if return_all:
        return sigma, angles, normals
    return sigma

__all__ = ["bootstrap_angles"]

"""
### Key points
* Draws **with replacement** to mimic sampling distribution.
* Resolves ± sign ambiguity by aligning every bootstrap draw to the
  direction of the first normal.
* Returns either just the **1-σ angle** or, optionally, every bootstrap
  angle and normal for richer diagnostics.
"""
