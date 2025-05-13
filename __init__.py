"""
mms_boundary_analysis
=====================

A lightweight, *batteries-included* toolkit for analysing
**Magnetospheric Multiscale (MMS)** string-of-pearls boundary crossings.

High-level features
-------------------
* **Download** → **detect** → **classify** → **CSV / Parquet** pipeline
  (see :pyfunc:`mms_boundary_analysis.events.run_pipeline`).
* Dual normal estimation (single-SC MVA and four-SC timing) with
  bootstrap error bars.
* ΔN distance time-series (local & Shue-98 model) and quick-look plots.
* Command-line wrapper::

      python -m mms_boundary_analysis.cli run …

Lazy import design
------------------
Heavy dependencies (NumPy, Matplotlib, *pyspedas*) are only imported the
first time the corresponding sub-module is actually used, keeping the
package import itself snappy.

Shortcuts
---------
The most common helpers are re-exported at top level:

* :pyfunc:`load_mms_data` – CDF fetch & clip
* :pyfunc:`run_pipeline` – one-liner end-to-end analysis
* :pyfunc:`plot_timeseries`, :pyfunc:`plot_normals` – quick-look figures
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

__version__ = "0.1.0"

# ---------------------------------------------------------------------
# Lazy-import machinery ------------------------------------------------
# ---------------------------------------------------------------------
_SUBMODULES = {
    # public_name      (module path, attr in that module)
    "load_mms_data":   ("io.loader",    "load_mms_data"),
    "run_pipeline":    ("events",       "run_pipeline"),
    "plot_timeseries": ("visual.timeseries",   "plot_timeseries"),
    "plot_normals":    ("visual.normals_plot", "plot_normals"),
    "get_omni_ctx":    ("io.omni",      "get_context"),
}

def __getattr__(name: str) -> Any:  # noqa: D401  (simple function)
    """
    Dynamically import heavy sub-modules on first access.
    """
    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_path, attr = _SUBMODULES[name]
    module: ModuleType = import_module(f".{mod_path}", __name__)
    value = getattr(module, attr)
    globals()[name] = value     # cache for next time
    return value


# ---------------------------------------------------------------------
# Re-export key names for *from mms_boundary_analysis import \* *
# ---------------------------------------------------------------------
__all__ = ["__version__"] + list(_SUBMODULES.keys())
