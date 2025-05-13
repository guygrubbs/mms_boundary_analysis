"""
mms_boundary_analysis.visual
============================

Light-weight Matplotlib helpers for *quick-look* plots produced by the
analysis pipeline.

Sub-modules
-----------

timeseries.py
    ΔN versus time: shows **local** and **model** (Shue-98) distances
    for all four spacecraft with event markers.

normals_plot.py
    Rose / quiver diagram of boundary normals (local vs. reference).

imf_scatter.py
    Scatter plot of IMF clock or cone angle **versus** event category
    (colour-coded), useful for trend inspection.

Public shortcuts
----------------
The most common routines are re-exported at package level so user code
can simply write::

    from mms_boundary_analysis.visual import plot_timeseries, plot_normals

without diving into the deeper module path.
"""

from __future__ import annotations
from importlib import import_module

# ---------------------------------------------------------------------
# Lazy imports keep heavy matplotlib dependencies out of the critical
# path when plotting is not requested.
# ---------------------------------------------------------------------
_timeseries   = import_module(".timeseries",   __package__)   # noqa: E402
_normals_plot = import_module(".normals_plot", __package__)   # noqa: E402
_imf_scatter  = import_module(".imf_scatter",  __package__)   # noqa: E402

# ---------------------------------------------------------------------
# Re-export public callables
# ---------------------------------------------------------------------
plot_timeseries = _timeseries.plot_timeseries
plot_normals    = _normals_plot.plot_normals
plot_imf_scatter = _imf_scatter.plot_imf_scatter

__all__ = [
    "plot_timeseries",
    "plot_normals",
    "plot_imf_scatter",
]
