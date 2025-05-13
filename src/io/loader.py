"""
src/io/loader.py
================

Thin wrapper around *pyspedas* / *pytplot* that fetches MMS spacecraft
data **and** returns it *already clipped* to the global analysis
interval defined in :pymod:`mms_boundary_analysis.config`.

Only two public symbols are exported:

* :func:`load_mms_data` – download MEC, FGM, and (optionally) FPI/HPCA
  CDFs for the requested probes, returning a dictionary of NumPy
  arrays (native cadence preserved).

* :func:`clip_window`   – utility to slice any ``(time, value)`` pair
  to the [start, stop] window, handling both POSIX‐seconds floats and
  ``numpy.datetime64`` grids.

The rest of the package relies on these helpers; **no other module
should talk to pyspedas / pytplot directly**.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
from pyspedas import mms
from pytplot import get_data

from .. import config

# ---------------------------------------------------------------------
# Utility: robust tplot getter
# ---------------------------------------------------------------------
def _try_get(name: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Return ``(time, values)`` or ``(None, None)`` if variable absent."""
    data = get_data(name)
    return (None, None) if data is None else data


# ---------------------------------------------------------------------
# clip_window
# ---------------------------------------------------------------------
def clip_window(
    t: np.ndarray,
    arr: np.ndarray,
    t0: np.datetime64,
    t1: np.datetime64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a time/value pair to *exactly* the interval [t0, t1].

    Parameters
    ----------
    t, arr
        1-D arrays of equal length. ``t`` may be either
        POSIX‐seconds floats (as used by *tplot*) **or**
        ``numpy.datetime64`` timestamps.
    t0, t1
        Window edges (datetime64 ns preferred).

    Returns
    -------
    (t_clipped, arr_clipped)
    """
    if t is None or arr is None:
        return (np.array([]), np.array([]))

    if np.issubdtype(t.dtype, np.floating):
        lo = (t0 - config.EPOCH64).astype('timedelta64[s]').astype(float)
        hi = (t1 - config.EPOCH64).astype('timedelta64[s]').astype(float)
        mask = (t >= lo) & (t <= hi)
    else:  # datetime64 grid
        mask = (t >= t0) & (t <= t1)

    return t[mask], arr[mask]


# ---------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------
def load_mms_data(
    trange: List[str] | None = None,
    probes: List[str] | None = None,
    *,
    include_fpi: bool = True,
    include_hpca: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Download & return MMS data for *probes* within *trange*.

    The function is **idempotent** – if CDFs already exist in the
    pyspedas cache, nothing is re-downloaded.

    Parameters
    ----------
    trange
        ``['YYYY-MM-DD/hh:mm:ss', 'YYYY-MM-DD/hh:mm:ss']``.
        If *None*, uses :pydata:`config.DEFAULT_START/STOP`.
    probes
        List like ``['1','2','3','4']``.  If *None* use
        :pydata:`config.PROBES`.
    include_fpi, include_hpca
        Toggle loading of FPI moments and HPCA moments.

    Returns
    -------
    dict
        ``data['mms1']['tpos']`` etc.  Keys:

        * ``tpos``, ``pos``  – MEC position (km, GSE)
        * ``tvi``, ``Vi``    – FPI DIS bulk velocity (kps, GSE)

    Notes
    -----
    *Native cadences are preserved.*  No resampling is performed.
    """
    if trange is None:
        trange = [
            config.DEFAULT_START.strftime('%Y-%m-%d/%H:%M:%S'),
            config.DEFAULT_STOP.strftime('%Y-%m-%d/%H:%M:%S'),
        ]
    if probes is None:
        probes = config.PROBES

    # ---- MEC & FGM always needed ------------------------------------
    mms.mec(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)
    mms.fgm(trange=trange, probe=probes, data_rate='srvy', level='l2', notplot=False)

    # ---- FPI moments -------------------------------------------------
    if include_fpi:
        mms.fpi(
            trange=trange,
            probe=probes,
            data_rate='fast',
            level='l2',
            datatype=['dis-moms', 'des-moms'],
            notplot=False,
        )
        # fallback: MMS-4 often lacks DES L2
        if get_data('mms4_des_numberdensity_fast') is None:
            mms.fpi(
                trange=trange,
                probe='4',
                data_rate='fast',
                level='ql',
                datatype='des-moms',
                notplot=False,
            )

    # ---- HPCA --------------------------------------------------------
    if include_hpca:
        for p in probes:
            mms.hpca(
                trange=trange,
                probe=p,
                data_rate='fast',
                level='l2',
                datatype='moments',
                notplot=True,
            )

    # -----------------------------------------------------------------
    # Package results
    # -----------------------------------------------------------------
    t0 = np.datetime64(config.DEFAULT_START)
    t1 = np.datetime64(config.DEFAULT_STOP)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for p in probes:
        sid = f"mms{p}"

        tp, pos = _try_get(f'{sid}_mec_r_gse')
        tv, Vi  = _try_get(f'{sid}_dis_bulkv_gse_fast')

        tp, pos = clip_window(tp, pos, t0, t1)
        tv, Vi  = clip_window(tv, Vi,  t0, t1)

        if tp.size == 0 or tv.size == 0:
            raise RuntimeError(f"{sid}: essential variables missing or out of window.")

        out[sid] = {
            'tpos': tp,
            'pos':  pos,
            'tvi':  tv,
            'Vi':   Vi,
        }

    return out
