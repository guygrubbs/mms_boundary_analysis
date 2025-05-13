"""
mms_boundary_analysis.visual.normals_plot
=========================================

3-D **quiver / rose** diagram of boundary‐normal vectors.

Purpose
-------
Visual sanity-check:

* Do the per-event MVA normals cluster tightly?  
* How far do they deviate from the reference (timing) normal?

Public helper
-------------
``plot_normals(normals_by_sc, /, *,
               ref_normal=None, sphere=True,
               title=None, ax=None)``

Parameters
~~~~~~~~~~
normals_by_sc : dict
    ``{ sc_id: [ n̂0, n̂1, … ] }`` – each *n̂* is a length-3 sequence.
ref_normal : array-like or ``None``
    If supplied, plotted as a thick black arrow for comparison.
sphere : bool, default ``True``
    Draws a translucent unit sphere for visual context.
title : str or ``None``
    Figure title; auto-generated if omitted.
ax : ``mpl_toolkits.mplot3d.axes3d.Axes3D`` or ``None``
    Existing 3-D axis; a new figure/axis is created otherwise.

Returns
~~~~~~~
``(fig, ax)`` – so callers can tweak or save.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect import)

# colour per MMS probe
_SC_COL = dict(mms1="tab:red", mms2="tab:blue",
               mms3="tab:green", mms4="tab:purple")


# ---------------------------------------------------------------------
# main plotting helper
# ---------------------------------------------------------------------
def plot_normals(
    normals_by_sc: Dict[str, List[Sequence[float]]],
    /,
    *,
    ref_normal: Sequence[float] | None = None,
    sphere: bool = True,
    title: str | None = None,
    ax=None
) -> Tuple[plt.Figure, "Axes3D"]:
    """
    Plot per-event normals in 3-D.

    See module docstring for parameter details.
    """
    if ax is None:
        fig = plt.figure(figsize=(6.5, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Draw unit sphere for context
    if sphere:
        u, v = np.mgrid[0 : 2 * np.pi : 60j, 0 : np.pi : 30j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_surface(xs, ys, zs, color="lightgrey", alpha=0.15,
                        zorder=0, linewidth=0)

    # Plot normals
    for sc, n_list in normals_by_sc.items():
        n_arr = np.asarray(n_list, dtype=float)
        if n_arr.size == 0:
            continue
        # Each arrow base at origin
        ax.quiver(
            np.zeros(len(n_arr)), np.zeros(len(n_arr)), np.zeros(len(n_arr)),
            n_arr[:, 0], n_arr[:, 1], n_arr[:, 2],
            color=_SC_COL.get(sc, "k"), length=1.0, normalize=True,
            linewidth=1, alpha=0.8, label=f"{sc.upper()}"
        )

    # Reference normal
    if ref_normal is not None:
        n_ref = np.asarray(ref_normal, dtype=float)
        n_ref = n_ref / np.linalg.norm(n_ref)
        ax.quiver(
            0, 0, 0, n_ref[0], n_ref[1], n_ref[2],
            color="k", linewidth=2.5, length=1.1, normalize=True,
            label="reference"
        )

    # Axis cosmetics
    ax.set_box_aspect([1, 1, 1])  # equal aspect
    lim = 1.2
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=35)
    if title is None:
        title = "Boundary normals – MVA vs reference"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    return fig, ax


__all__ = ["plot_normals"]
