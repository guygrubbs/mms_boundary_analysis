"""
mms_boundary_analysis.io
========================
Input / output helper hub for the MMS boundary-analysis package.

Sub-modules
-----------

loader.py
    Thin wrappers around *pyspedas* / *pytplot*.  Downloads MEC, FGM,
    FPI, HPCA CDFs **and** clips every variable to the global time
    window while preserving native cadence.

omni.py
    Retrieves 1-min (or higher) OMNI solar-wind data and exposes
    helper ``get_context(t)``, returning a dictionary with Bz, By,
    clock / cone angle, dynamic pressure, *etc.*

writer.py
    Convenience functions for writing per-spacecraft CSV / Parquet /
    HDF5 files that conform to the column order defined in
    ``config.CSV_COLUMNS``.

shue_coeffs.json
    Coefficients for the Shue (1998) magnetopause model—shared with
    ``distance.shue``.

This ``__init__`` file re-exports the *most common* top-level helpers
so callers can simply do::

    from mms_boundary_analysis.io import load_mms_data, get_omni_ctx, save_csv
"""

from __future__ import annotations
from pathlib import Path
from importlib import import_module

from ..config import DATA_DIR

# ---------------------------------------------------------------------
# Lazy imports keep heavy dependencies (pyspedas) from slowing down
# package import if a user only wants config or plotting utilities.
# ---------------------------------------------------------------------
_loader = import_module('.loader',   __package__)   # noqa: E402
_omni   = import_module('.omni',     __package__)   # noqa: E402
_writer = import_module('.writer',   __package__)   # noqa: E402

# -------- public shortcuts -------------------------------------------
load_mms_data = _loader.load_mms_data       # (trange, probes) → dict
clip_window   = _loader.clip_window

get_omni_ctx  = _omni.get_context           # (datetime or np.datetime64) → dict

save_csv      = _writer.save_csv            # (DataFrame, filename)
save_parquet  = _writer.save_parquet

# Ensure the data directory exists (safe, idempotent)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

__all__ = [
    "load_mms_data",
    "clip_window",
    "get_omni_ctx",
    "save_csv",
    "save_parquet",
    "DATA_DIR",
]
