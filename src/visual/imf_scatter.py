"""
mms_boundary_analysis.visual.imf_scatter
========================================

Quick-look **scatter plot** that places each detected boundary event in
the plane *(IMF orientation metric, event category)*.

*Metric* options
----------------
* ``'clock'``  – IMF clock angle (°)  … default
* ``'cone'``   – IMF cone  angle (°)
* Any other **numeric** key present in the event dictionaries
  (e.g. ``'Pdyn_nPa'``).

Colour-coding
-------------
By default points are coloured by event ``'category'`` as assigned by
:pyfunc:`mms_boundary_analysis.detect.classify.classify`.  You may pass
``color_by='cross_type'`` or any other categorical key present in the
events.

Public helper
-------------
``plot_imf_scatter(events_by_sc, /, *,
                   metric='clock', color_by='category',
                   title=None, ax=None)``

Returns ``(fig, ax)`` so that the caller can further customise or save
the figure.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Consistent colour palette across plots
_CAT_COL = {
    "MP full": "tab:red",
    "MP ion-skim": "tab:orange",
    "EDR": "tab:blue",
    "plume": "tab:green",
    "unknown": "grey",
    "cross": "tab:purple",
    "skim": "tab:brown",
}

# Small shift to separate categories on y-axis
_Y_JITTER = dict()


# ---------------------------------------------------------------------
def _collect_points(
    events_by_sc: Dict[str, List[dict]],
    metric: str,
    color_by: str,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Flatten event dict → arrays of x, y (category index), colours."""
    xs, ys, cols = [], [], []

    # create stable category order
    categories = []
    for lst in events_by_sc.values():
        for ev in lst:
            cat = ev.get(color_by, "unknown")
            if cat not in categories:
                categories.append(cat)

    # mapping cat→index for y positioning
    cat_to_y = {c: i for i, c in enumerate(categories)}

    # gather points
    for sc, lst in events_by_sc.items():
        for ev in lst:
            x = ev.get(metric if metric not in ("clock", "cone") else f"{metric}_deg")
            if x is None or np.isnan(x):
                continue
            cat = ev.get(color_by, "unknown")
            xs.append(float(x))
            # store category index with small jitter to avoid overlap
            jitter = _Y_JITTER.setdefault(cat, np.random.normal(scale=0.04))
            ys.append(cat_to_y[cat] + jitter)
            cols.append(_CAT_COL.get(cat, "black"))

    return np.asarray(xs), np.asarray(ys), cols, categories


# ---------------------------------------------------------------------
# public helper
# ---------------------------------------------------------------------
def plot_imf_scatter(
    events_by_sc: Dict[str, List[dict]],
    /,
    *,
    metric: str = "clock",
    color_by: str = "category",
    title: str | None = None,
    ax=None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter IMF metric versus event category.

    Parameters
    ----------
    events_by_sc : dict
        Nested events structure from the pipeline.
    metric : str, default 'clock'
        X-axis quantity – 'clock', 'cone', or any numeric key in events.
    color_by : str, default 'category'
        Event dict key used for colour-grouping.
    title : str or None
        Figure title; generated automatically if None.
    ax : matplotlib.axes.Axes or None
        Existing axis; new figure is created if None.

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
    else:
        fig = ax.figure

    xs, ys, cols, cats = _collect_points(events_by_sc, metric, color_by)
    ax.scatter(xs, ys, c=cols, s=40, alpha=0.8, edgecolor="k", linewidth=0.4)

    # y-axis ticks / labels
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_ylim(-1, len(cats) - 0.5)

    # x-axis label
    if metric in ("clock", "cone"):
        ax.set_xlabel(f"IMF {metric.capitalize()} angle (deg)")
    else:
        ax.set_xlabel(metric)

    ax.set_ylabel(color_by.replace("_", " ").title())

    ax.grid(True, axis="x", alpha=0.3)
    if title is None:
        title = f"{metric.capitalize()} vs {color_by.replace('_', ' ')}"
    ax.set_title(title)

    # Construct legend only for categories present
    handles = []
    labels = []
    for cat in cats:
        if cat in _CAT_COL:
            patch = plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=_CAT_COL[cat], markeredgecolor="k",
                               markersize=8, label=cat)
            handles.append(patch)
            labels.append(cat)
    ax.legend(handles, labels, fontsize=8, frameon=True, loc="best")

    return fig, ax


__all__ = ["plot_imf_scatter"]
