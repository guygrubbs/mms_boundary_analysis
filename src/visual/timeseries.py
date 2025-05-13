"""
mms_boundary_analysis.visual.timeseries
=======================================

Quick-look plot of **distance-to-boundary** versus time for each
spacecraft.

The routine visualises two curves per probe:

* **ΔN_local**   – signed distance along the *event* normal  
  (computed in :pyfunc:`mms_boundary_analysis.distance.local.delta_n_local`)
* **ΔN_model**   – signed distance to the Shue-98 model surface  
  (optional; give ``show_model=False`` to hide)

Event timestamps supplied through *events_by_sc* are annotated with
category-coded markers.

Public function
---------------

``plot_timeseries(dseries, events_by_sc, /, *,
                  show_model=True, title=None, ax=None)``

* *dseries* is a dict keyed by spacecraft:

    ``{ 'mms1': {'time': np.array, 'local': np.array,
                              'model': np.array}, … }``

  Times can be POSIX seconds or ``numpy.datetime64``; distances in km.

* *events_by_sc* is the nested list returned by
  :pyfunc:`mms_boundary_analysis.events.run_pipeline`.

Returns the Matplotlib ``fig, ax`` so callers can tweak or save.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ..config import COLUMNS


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
EPOCH_1970 = mdates.date2num(np.datetime64("1970-01-01T00:00:00Z").astype("datetime64[s]"))


def _to_mpl(t_arr: np.ndarray) -> np.ndarray:
    """
    Convert POSIX seconds **or** numpy.datetime64 array → Matplotlib float.
    """
    t_arr = np.asarray(t_arr)
    if np.issubdtype(t_arr.dtype, np.floating):          # POSIX seconds
        return t_arr / 86400.0 + EPOCH_1970
    if np.issubdtype(t_arr.dtype, "datetime64"):
        sec = t_arr.astype("datetime64[s]").astype(float)
        return sec / 86400.0 + EPOCH_1970
    raise TypeError("Unsupported time array dtype for _to_mpl()")


_SC_COL = dict(mms1="tab:red", mms2="tab:blue",
               mms3="tab:green", mms4="tab:purple")

_EVT_MK = dict(
    **{ "MP full": "o", "MP ion-skim": "^", "EDR": "s",
        "plume": "D", "unknown": "x" }
)

# ---------------------------------------------------------------------
# main plotting helper
# ---------------------------------------------------------------------
def plot_timeseries(
    dseries: Dict[str, Dict[str, np.ndarray]],
    events_by_sc: Dict[str, List[dict]],
    /,
    *,
    show_model: bool = True,
    title: str | None = None,
    ax=None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ΔN versus time for each spacecraft with event markers.

    Parameters
    ----------
    dseries : dict
        ``{ sc: {'time': arr, 'local': arr, 'model': arr?} }``.
    events_by_sc : dict
        Nested structure from the main analysis pipeline.
    show_model : bool, default True
        If *False* hides the Shue-model curve.
    title : str or None
        Plot title; auto-generated if None.
    ax : matplotlib.axes.Axes or None
        Existing axis to draw on; new figure is created otherwise.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    for sc, ds in dseries.items():
        t_mpl = _to_mpl(ds["time"])
        ax.plot(t_mpl, ds["local"],
                color=_SC_COL.get(sc, "k"),
                lw=1.5, label=f"{sc.upper()}  ΔN local")
        if show_model and "model" in ds and np.isfinite(ds["model"]).any():
            ax.plot(t_mpl, ds["model"],
                    color=_SC_COL.get(sc, "k"),
                    lw=1, ls="--", alpha=0.7,
                    label=f"{sc.upper()}  ΔN model")

        # --- overlay event markers -----------------------------------
        for ev in events_by_sc.get(sc, []):
            t_evt = _to_mpl(np.array([ev["time"]]))[0]
            y_evt = np.interp(t_evt, t_mpl, ds["local"])
            cat = ev.get("category", "unknown")
            ax.scatter(t_evt, y_evt,
                       marker=_EVT_MK.get(cat, "x"),
                       s=70, color=_SC_COL.get(sc, "k"),
                       edgecolor="k", zorder=6,
                       label=cat if cat not in ax.get_legend_handles_labels()[1] else "")

    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_ylabel("ΔN  (km)")
    ax.set_xlabel("UT")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Distance to Boundary – MMS string-of-pearls"
    ax.set_title(title)

    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()

    return fig, ax


__all__ = ["plot_timeseries"]
