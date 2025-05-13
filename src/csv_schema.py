"""
mms_boundary_analysis.csv_schema
================================

Canonical column order and dtype map used for *every* CSV / Parquet
written by :pymod:`mms_boundary_analysis.io.writer`.

Having a single place for the schema means:

* Writer and downstream ingestion (e.g. plotting notebooks) never drift
  out of sync.
* We avoid magic strings scattered throughout the codebase.

If you add or rename a column, change it **here** and nowhere else.
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Primary column list  (kept in same order as quick-look files)
# ---------------------------------------------------------------------
COLUMNS: list[str] = [
    "iso_time",               # ISO-8601 UTC string
    "delta_N_local_km",       # (r − r0)·N̂  from event MVA normal
    "delta_N_local_ref_km",   # same but w.r.t. reference (global) normal
    "delta_N_model_km",       # Shue-98 model distance
    "N_angle_ref_deg",        # angle between local & reference normals
    "Bz_nT", "By_nT",         # upstream IMF (GSM/GSE)
    "clock_deg", "cone_deg",  # orientation metrics
    "Vsw_kms", "Pdyn_nPa",    # solar-wind speed & dynamic pressure
    "category",               # MP full, EDR, …
    "cross_type",             # cross / skim  (optional)
    "event_id",               # MMS#_YYYYMMDDThhmmss
]

# ---------------------------------------------------------------------
# Recommended pandas dtypes (for write_parquet).  Feel free to tweak.
# ---------------------------------------------------------------------
DTYPES: Dict[str, str | type] = {
    "iso_time":               "string[pyarrow]",
    "delta_N_local_km":       "float64",
    "delta_N_local_ref_km":   "float64",
    "delta_N_model_km":       "float64",
    "N_angle_ref_deg":        "float64",
    "Bz_nT":                  "float32",
    "By_nT":                  "float32",
    "clock_deg":              "float32",
    "cone_deg":               "float32",
    "Vsw_kms":                "float32",
    "Pdyn_nPa":               "float32",
    "category":               "category",
    "cross_type":             "category",
    "event_id":               "string[pyarrow]",
}

# Provide a ready-made empty DataFrame for convenience
EMPTY_DF: pd.DataFrame = pd.DataFrame({c: pd.Series(dtype=DTYPES.get(c, "float64"))
                                       for c in COLUMNS})

__all__ = ["COLUMNS", "DTYPES", "EMPTY_DF"]
