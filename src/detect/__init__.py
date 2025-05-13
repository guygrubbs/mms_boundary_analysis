"""
mms_boundary_analysis.detect
============================

Boundary–crossing **event detection** utilities.

Sub-modules
-----------

flip_detector.py
    Primary algorithm that scans 4-second MMS time series for
    *candidate* crossings based on a concurrent sharp rotation of
    the magnetic field **and** a density drop (or rise) in one or
    more plasma species.

prune.py
    Groups neighbouring candidates, enforces a minimum separation
    (config.MIN_SEPARATION_SEC), and keeps the highest-score event
    within each group.

classify.py
    Assigns a high-level **category** to every pruned event
    (“MP full”, “MP skim”, “EDR”, “plume”, …) using the drop
    fractions of H<sup>+</sup>, e<sup>−</sup>, O<sup>+</sup>, He<sup>+</sup>,
    plus rotation magnitude.

Public shortcuts
----------------
Importing the sub-package re-exports the most common helpers so that
callers can write simply::

    from mms_boundary_analysis.detect import find_candidates, prune_events

instead of diving into the deeper module path.
"""

from __future__ import annotations
from importlib import import_module

# Lazy imports so heavy NumPy / SciPy code loads only when needed
_flip  = import_module('.flip_detector', __package__)   # noqa: E402
_prune = import_module('.prune',         __package__)   # noqa: E402
_class = import_module('.classify',      __package__)   # noqa: E402

# ---- public shortcuts --------------------------------------------------------
find_candidates = _flip.find_candidates          # raw B-flip + density drop list
prune_events    = _prune.prune_candidates        # apply min-separation, scoring
classify_event  = _class.classify                # add high-level category label

__all__ = [
    "find_candidates",
    "prune_events",
    "classify_event",
]
