"""
mms_boundary_analysis.io.omni
=============================

Download / cache upstream solar-wind & IMF parameters from **OMNIWeb**
(via *pyspedas*) and expose a single convenience helper::

    >>> from mms_boundary_analysis.io import get_omni_ctx
    >>> ctx = get_omni_ctx(np.datetime64('2019-01-27T12:23:15Z'))
    >>> ctx['clock_deg'], ctx['Pdyn_nPa']
    (174.2, 1.9)

The first call within a session triggers a download (5-second to
1-minute cadence, depending on availability) for the global
time-interval defined in :pymod:`mms_boundary_analysis.config`.  The
data are cached in *data/omni.pkl* so subsequent runs start instantly.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .. import config

try:
    from pyspedas import omni
    from pytplot   import get_data
except ImportError as exc:  # allow docs to build w/o heavy deps
    raise ImportError("pyspedas + pytplot required for mms_boundary_analysis.io.omni") from exc


# ---------------------------------------------------------------------
# File locations & globals
# ---------------------------------------------------------------------
_CACHE_FILE: Path = config.DATA_DIR / "omni.pkl"
_COLUMNS = [
    "utc",
    "Bx_nT", "By_nT", "Bz_nT",
    "Vsw_kms", "Nsw_cm3",
    "Pdyn_nPa", "clock_deg", "cone_deg"
]

_OMNI_DF: Optional[pd.DataFrame] = None   # filled lazily


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _calc_angles(row: pd.Series) -> pd.Series:
    """Return clock & cone angle for a row (Bx,By,Bz)."""
    Bx, By, Bz = row[["Bx_nT", "By_nT", "Bz_nT"]]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2) or np.nan
    clock = np.degrees(np.arctan2(abs(By), Bz)) if Bmag else np.nan
    cone  = np.degrees(np.arccos(Bx / Bmag))    if Bmag else np.nan
    return pd.Series({"clock_deg": clock, "cone_deg": cone})


def _download_omni() -> pd.DataFrame:
    """Fetch OMNI data for the global time window, cache to disk."""
    tr = [config.DEFAULT_START.strftime("%Y-%m-%d/%H:%M:%S"),
          config.DEFAULT_STOP .strftime("%Y-%m-%d/%H:%M:%S")]
    omni_vars = ["bx_gsm","by_gsm","bz_gsm","flow_speed","density","pressure"]
    omni.data(trange=tr, varnames=omni_vars, notplot=False)

    # pytplot returns separate tplot vars; convert to DataFrame
    t = get_data("bx_gsm")[0]          # seconds since 1970
    data = {
        "utc":      pd.to_datetime(t, unit="s", utc=True),
        "Bx_nT":    get_data("bx_gsm")[1],
        "By_nT":    get_data("by_gsm")[1],
        "Bz_nT":    get_data("bz_gsm")[1],
        "Vsw_kms":  get_data("flow_speed")[1],
        "Nsw_cm3":  get_data("density")[1],
        "Pdyn_nPa": get_data("pressure")[1],
    }
    df = pd.DataFrame(data).set_index("utc")
    df[["clock_deg", "cone_deg"]] = df.apply(_calc_angles, axis=1)
    df.to_pickle(_CACHE_FILE)
    return df


def _load_cache() -> pd.DataFrame:
    if _CACHE_FILE.exists():
        return pd.read_pickle(_CACHE_FILE)
    return _download_omni()


def _ensure_loaded() -> pd.DataFrame:
    global _OMNI_DF
    if _OMNI_DF is None:
        _OMNI_DF = _load_cache().sort_index()
    return _OMNI_DF


# ---------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------
def get_context(t: "datetime|np.datetime64|float") -> Dict[str, float]:
    """
    Return nearest-sample IMF & solar-wind context for time *t*.

    Parameters
    ----------
    t : datetime | numpy.datetime64 | float
        UTC moment for which to retrieve OMNI data.  If *float* is
        passed, it is interpreted as POSIX seconds.

    Returns
    -------
    dict
        Keys: ``Bx_nT, By_nT, Bz_nT, Vsw_kms, Nsw_cm3, Pdyn_nPa,
        clock_deg, cone_deg``.
    """
    if isinstance(t, (int, float)):
        t_utc = datetime.fromtimestamp(float(t), tz=timezone.utc)
    elif isinstance(t, np.datetime64):
        t_utc = pd.to_datetime(str(t)).tz_convert("UTC")
    elif isinstance(t, datetime):
        t_utc = t.astimezone(timezone.utc)
    else:
        raise TypeError("Unsupported time type for get_context")

    df = _ensure_loaded()
    try:
        row = df.iloc[df.index.get_indexer([t_utc], method="nearest")[0]]
    except (KeyError, IndexError):
        raise ValueError("Requested time outside cached OMNI interval")

    return {k: float(row[k]) for k in _COLUMNS if k != "utc"}
