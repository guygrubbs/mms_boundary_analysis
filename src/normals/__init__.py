"""
mms_boundary_analysis.normals
=============================

Algorithms that estimate magnetopause (or other boundary) **normal
vectors** and their uncertainties.

Sub-modules
-----------

mva.py
    Classic single-spacecraft **Minimum Variance Analysis** (MVA).

timing.py
    Four-spacecraft **timing method** – solves for the planar normal
    and boundary-speed that best explain the observed crossing offsets.

bootstrap.py
    Simple bootstrap resampling utilities for propagating magnetic-field
    noise into normal-direction (Δθ) uncertainty.

Public shortcuts
----------------
Importing this sub-package exposes the most common helpers directly::

    from mms_boundary_analysis.normals import mva_normal, timing_normal, bootstrap_angle

so callers do not need to know the exact module split.
"""

from __future__ import annotations

from importlib import import_module

# Lazy import so that heavy numpy operations are only loaded when needed
_mva       = import_module('.mva',       __package__)
_timing    = import_module('.timing',    __package__)
_bootstrap = import_module('.bootstrap', __package__)

# Re-export the key callables
mva_normal       = _mva.mva_normal
timing_normal    = _timing.timing_normal
bootstrap_angles = _bootstrap.bootstrap_angles

__all__ = [
    "mva_normal",
    "timing_normal",
    "bootstrap_angles",
]
