"""
mms_boundary_analysis.imf_context
=================================

Solar-wind and IMF **context utilities**.

The sub-package currently hosts a single module:

orientation.py
    Computes *clock angle*, *cone angle*, and convenient derived
    quantities (e.g. Alfvén Mach number when density is provided) from
    OMNI GSM/GSE vectors supplied by
    :pyfunc:`mms_boundary_analysis.io.omni.get_context`.

The design mirrors other packages in this library: we expose the most
frequently-used helper at sub-package level so downstream code can
write::

    from mms_boundary_analysis.imf_context import get_angles

instead of specifying the deeper path.

Future expansion (e.g. coupling-function calculators) can live here
without changing the public import surface.
"""

from __future__ import annotations
from importlib import import_module

# Lazy import keeps heavy dependencies (NumPy) out of the critical path
_orientation = import_module(".orientation", __package__)   # noqa: E402

# Public shortcut
get_angles = _orientation.get_angles

__all__ = ["get_angles"]
