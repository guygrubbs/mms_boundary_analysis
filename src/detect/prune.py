"""
mms_boundary_analysis.detect.prune
==================================

Take the *raw* candidate list produced by
:meth:`mms_boundary_analysis.detect.flip_detector.find_candidates`
and pare it down to a clean **events list** for each spacecraft.

Rules
-----

1. **Minimum temporal separation**  
   Two candidates closer than ``config.MIN_SEPARATION_SEC`` are assumed
   to belong to the *same* physical boundary.  Only one survives.

2. **Scoring**  
   When candidates collide within that window the one with the
   **highest score** wins::

       score = w_rot · (rot / 180°)  +  (1 − w_rot) · max_drop

   where *rot* is the magnetic rotation (deg) and *max_drop* is the
   largest fractional density change among all species.  The weighting
   factor ``w_rot`` defaults to *0.6* (can be overridden).

Public function
---------------

``prune_candidates(cands_by_sc, /, *, min_sep=None, w_rot=0.6)``

returns the same dictionary structure but with the inferior “dupes”
removed and a new key ``'score'`` added to each surviving event.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .. import config


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _score(cand: dict, *, w_rot: float) -> float:
    """Combined rotation + density score (0–1)."""
    rot_norm = cand["rot_deg"] / 180.0          # 0–1
    drops = np.array(
        [v for v in cand["drops"].values() if not np.isnan(v)], dtype=float
    )
    max_drop = drops.max() if drops.size else 0.0
    return w_rot * rot_norm + (1.0 - w_rot) * max_drop


# ---------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------
def prune_candidates(
    cands_by_sc: Dict[str, List[dict]],
    *,
    min_sep: float | None = None,
    w_rot: float = 0.6,
) -> Dict[str, List[dict]]:
    """
    Remove duplicate candidates that are too close in time.

    Parameters
    ----------
    cands_by_sc : dict
        Output of ``find_candidates``.
    min_sep : float or None
        Minimum allowed separation between successive events **in
        seconds**.  ``None`` ⇒ use ``config.MIN_SEPARATION_SEC``.
    w_rot : float, default 0.6
        Weight for magnetic rotation (0 ≤ w_rot ≤ 1).

    Returns
    -------
    dict
        Same structure but pruned; each event dict gains a ``'score'``
        field for downstream sorting.
    """
    if not (0.0 <= w_rot <= 1.0):
        raise ValueError("w_rot must be between 0 and 1")

    gap = float(min_sep if min_sep is not None else config.MIN_SEPARATION_SEC)
    pruned: Dict[str, List[dict]] = {}

    for sc_id, cands in cands_by_sc.items():
        # compute score for each candidate
        for cand in cands:
            cand["score"] = _score(cand, w_rot=w_rot)

        # sort by time
        cands_sorted = sorted(cands, key=lambda c: c["time"])

        keeper: List[dict] = []
        for cand in cands_sorted:
            if not keeper:
                keeper.append(cand)
                continue

            # seconds between this and last kept
            dt = cand["time"] - keeper[-1]["time"]
            if dt >= gap:
                keeper.append(cand)
            else:
                # within exclusion window – keep the one with higher score
                if cand["score"] > keeper[-1]["score"]:
                    keeper[-1] = cand  # replace with the better one

        pruned[sc_id] = keeper

    return pruned


__all__ = ["prune_candidates"]
