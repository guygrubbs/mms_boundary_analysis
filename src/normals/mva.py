"""
mms_boundary_analysis.normals.mva
=================================

Single–spacecraft **Minimum Variance Analysis** (MVA).

The procedure (Sonnerup & Cahill 1967) finds the boundary-normal vector
as the direction of *minimum* magnetic-field variance in a selected
time window.

Steps
-----
1.  Subtract the mean from **B** → fluctuations **δB**.
2.  Form the 3 × 3 covariance matrix **C = cov(δB)**.
3.  Solve **C v = λ v** – eigenvalues λ sorted λ₁ ≥ λ₂ ≥ λ₃.
4.  The eigenvector for λ₃ (smallest) is the **normal n̂**  
    (λ₂ ↔ **M̂**, λ₁ ↔ **L̂**).
5.  Quality check: λ₂ / λ₃ ≫ 1 ⇒ well-defined normal.

With *return_full=True* the routine also returns eigenvalues and the
full eigenvector matrix so callers can inspect λ-ratios.

Example
-------
```python
import numpy as np
from mms_boundary_analysis.normals import mva_normal

# Synthetic rotation: B rotates in the X-Y plane
t  = np.linspace(0, 2*np.pi, 400)
Bx = np.sin(t);  By = np.cos(t);  Bz = 0.05*np.ones_like(t)
n, lambdas, vecs = mva_normal(np.c_[Bx, By, Bz], return_full=True)

print("Normal ≈", n.round(3))
print("λ-ratio λ₂/λ₃ =", lambdas[1] / lambdas[2])
````

"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------

# Main routine

# ---------------------------------------------------------------------

def mva_normal(B: np.ndarray, *, return_full: bool = False) -> "np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]":

    """
    Minimum variance analysis on a magnetic-field array.

    Parameters
    ----------
    B : ndarray of shape (N, 3)
        Magnetic-field samples (any units).
    return_full : bool, default ``False``
        If *True*, also return eigenvalues and eigenvectors.

    Returns
    -------
    n_hat : ndarray, shape (3,)
        Unit boundary-normal vector (minimum variance).
    lambdas : ndarray, shape (3,)   (only if *return_full*)
        Eigenvalues λ₁ ≥ λ₂ ≥ λ₃.
    vecs : ndarray, shape (3, 3)    (only if *return_full*)
        Eigenvectors as **columns** [L̂, M̂, N̂].
    Raises
    ------
    ValueError
        If *B* is not shape (N, 3) with N ≥ 4.
    """

    B = np.asarray(B, dtype=float)
    if B.ndim != 2 or B.shape[1] != 3 or B.shape[0] < 4:
        raise ValueError("B must have shape (N>=4, 3)")

    # Remove DC component
    B_fluct = B - B.mean(axis=0, keepdims=True)

    # 3×3 covariance matrix
    C = np.cov(B_fluct, rowvar=False, bias=True)

    # Eigen-decomposition (ascending λ)
    lambdas, vecs = np.linalg.eigh(C)

    # Sort descending → vecs[:,0]=L̂, vecs[:,1]=M̂, vecs[:,2]=N̂
    order = np.argsort(lambdas)[::-1]
    lambdas = lambdas[order]
    vecs = vecs[:, order]

    # Ensure right-handed LMN
    if np.linalg.det(vecs) < 0:
        vecs[:, 0] *= -1.0

    n_hat = vecs[:, 2] / np.linalg.norm(vecs[:, 2])

    if return_full:
        return n_hat, lambdas, vecs
    return n_hat

# ---------------------------------------------------------------------

# Helper

# ---------------------------------------------------------------------

def angle_between(n1: np.ndarray, n2: np.ndarray) -> float:
    """
    Angle in degrees between two vectors *n1* and *n2*.

    Parameters
    ----------
    n1, n2 : ndarray, shape (3,)
        Unit vectors.

    Returns
    -------
    float
    """
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    cosang = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

__all__ = ["mva_normal", "angle_between"]
