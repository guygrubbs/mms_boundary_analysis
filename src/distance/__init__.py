"""
mms_boundary_analysis.distance
==============================

Distance-to-boundary utilities.

Sub-modules
-----------

local.py
    *Local* method – signed distance **ΔN = (r − r₀) · n̂** where *r₀*
    is the spacecraft position at the current-sheet time.

shue.py
    Global method using the empirical **Shue et al. (1998)**
    magnetopause model.  Returns radial standoff distance and helpers
    for intersection along an arbitrary normal.

projection.py
    Generic line→surface intersection used by shue.py (and extendable
    to other analytic magnetopause or bow-shock models).

The `distance` package re-exports the most common helpers so callers
can simply write::

    from mms_boundary_analysis.distance import delta_n_local, shue_radius
"""

from __future__ import annotations
from importlib import import_module

# Lazy   imports keep heavy NumPy functions from loading until needed
_local      = import_module('.local',      __package__)   # noqa: E402
_shue       = import_module('.shue',       __package__)   # noqa: E402
_projection = import_module('.projection', __package__)   # noqa: E402

# ---- public shortcuts --------------------------------------------------------
delta_n_local   = _local.delta_n_local
shue_radius     = _shue.shue_radius
delta_n_shue    = _shue.delta_n_shue      # along event normal
project_along_n = _projection.project_along_normal

__all__ = [
    "delta_n_local",
    "shue_radius",
    "delta_n_shue",
    "project_along_n",
]
