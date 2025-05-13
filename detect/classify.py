"""
mms_boundary_analysis.detect.classify
=====================================

Light-weight *heuristic* classifier that assigns a human-readable
**category** to each pruned event.

The scheme is deliberately simple and transparent; refine it for your
own science questions.

Rules
-----

========  ============================================================
Category   Criteria (drop fractions are the species’ “pre − post / pre”
           computed in :pymod:`flip_detector`)
========  ============================================================
MP full    H⁺ drop ≥ 0.70 **and** e⁻ drop ≥ 0.70
MP ion-skim  H⁺ drop ≥ 0.50 **and** e⁻ drop < 0.50 (or NaN)
EDR        e⁻ drop ≥ 0.70 **and** H⁺ drop < 0.30 (or NaN)
plume      O⁺ or He⁺ drop ≥ 0.50 **and** H⁺ drop < 0.40
unknown    Anything else
========  ============================================================

Additionally, if the calling code provides the *layer thickness*
|ΔN| (km) the event is flagged as

* ``cross``  –  |ΔN| ≥ 500 km  
* ``skim``   –  |ΔN| < 500 km  (spacecraft grazes boundary without
  fully traversing it)

Public helpers
--------------

``classify(event, thickness_km=None, /, update=True)``  
Assigns ``event['category']`` and (optionally) ``event['cross_type']``  
and returns the category string.

``classify_all(events_by_sc, thickness_dict=None)``  
Loops over the nested structure returned by
:meth:`mms_boundary_analysis.detect.prune.prune_candidates`.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------
def _get_drop(ev: dict, sp: str) -> float:
    """Return drop fraction for species *sp* or NaN if missing."""
    val = ev["drops"].get(sp)
    return float("nan") if val is None else float(val)


# ---------------------------------------------------------------------
def classify(
    ev: dict,
    thickness_km: Optional[float] = None,
    /,
    *,
    update: bool = True,
) -> str:
    """
    Assign a high-level category label to *ev*.

    Parameters
    ----------
    ev : dict
        One event dictionary (must contain ``'drops'``).
    thickness_km : float or None
        Signed layer thickness |ΔN| (km); if supplied adds
        ``'cross_type'`` (“cross” or “skim”).
    update : bool, default ``True``
        If *True* the function **adds** the new keys to *ev*.

    Returns
    -------
    str
        Category label.
    """
    # pull species drops (may be NaN)
    h  = _get_drop(ev, "H+")
    e  = _get_drop(ev, "e-")
    he = _get_drop(ev, "He+")
    o  = _get_drop(ev, "O+")

    # ----- decision tree --------------------------------------------------
    if h >= 0.70 and e >= 0.70:
        cat = "MP full"
    elif h >= 0.50 and (e < 0.50 or e != e):          # e NaN → skim ion
        cat = "MP ion-skim"
    elif e >= 0.70 and (h < 0.30 or h != h):
        cat = "EDR"
    elif (he >= 0.50 or o >= 0.50) and (h < 0.40 or h != h):
        cat = "plume"
    else:
        cat = "unknown"

    cross_type = None
    if thickness_km is not None:
        cross_type = "skim" if abs(thickness_km) < 500.0 else "cross"

    # ----- attach to dict -------------------------------------------------
    if update:
        ev["category"] = cat
        if cross_type:
            ev["cross_type"] = cross_type

    return cat


# ---------------------------------------------------------------------
def classify_all(
    events_by_sc: Dict[str, List[dict]],
    thickness_dict: Optional[Dict[str, List[float]]] = None,
) -> None:
    """
    In-place classification of **all** events in the nested structure.

    Parameters
    ----------
    events_by_sc : dict
        ``{ sc_id: [event, …] }`` – mutated in place.
    thickness_dict : dict or None
        Optional nested structure with matched layer thicknesses
        ``{ sc_id: [thick0, thick1, …] }``.  If provided, index position
        must match *events_by_sc* order.
    """
    for sc_id, ev_list in events_by_sc.items():
        thicks = thickness_dict.get(sc_id) if thickness_dict else None
        for i, ev in enumerate(ev_list):
            thick = thicks[i] if thicks is not None and i < len(thicks) else None
            classify(ev, thickness_km=thick, update=True)


__all__ = ["classify", "classify_all"]
