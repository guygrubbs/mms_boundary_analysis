"""
tests/test_detector.py
======================

Minimal unit-tests for the detector stack.  These tests **do not** touch
PySPEDAS or real CDFs; they rely solely on the pure-Python helpers in
*detect.prune* and *detect.classify*.

Run with::

    $ pytest -q
"""

from __future__ import annotations

import numpy as np

from mms_boundary_analysis.detect.prune import prune_candidates
from mms_boundary_analysis.detect.classify import classify
from mms_boundary_analysis.detect.flip_detector import find_candidates
from mms_boundary_analysis import config


# ---------------------------------------------------------------------
# prune_candidates
# ---------------------------------------------------------------------
def test_prune_keeps_highest_score() -> None:
    """
    Two raw candidates closer than *min_sep* → only the higher-score
    event should survive.
    """
    raw = {
        "mms1": [
            # candidate A  (higher rotation ⇒ higher score)
            {
                "time": 0.0,            # seconds
                "rot_deg": 90.0,
                "drops": {"H+": 0.8, "e-": 0.8},
            },
            # candidate B  (lower rotation)
            {
                "time": 10.0,           # within the 30 s window
                "rot_deg": 60.0,
                "drops": {"H+": 0.6, "e-": 0.6},
            },
        ]
    }

    pruned = prune_candidates(raw, min_sep=30, w_rot=0.6)
    kept = pruned["mms1"]

    assert len(kept) == 1, "expected one surviving event"
    assert kept[0]["rot_deg"] == 90.0, "highest-score event should be kept"
    # score should be attached
    assert "score" in kept[0]


# ---------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------
def test_classify_categories_and_cross_type() -> None:
    """
    Verify the heuristic rules in *detect.classify*.
    """
    # --- MP full (cross) --------------------------------------------
    ev_full = {
        "drops": {"H+": 0.8, "e-": 0.8, "He+": 0.0, "O+": 0.0}
    }
    cat_full = classify(ev_full, 600)   # crosses boundary
    assert cat_full == "MP full"
    assert ev_full["cross_type"] == "cross"

    # --- ion-skim ----------------------------------------------------
    ev_skim = {
        "drops": {"H+": 0.55, "e-": 0.20, "He+": 0.1, "O+": 0.1}
    }
    cat_skim = classify(ev_skim, 200)   # skim
    assert cat_skim == "MP ion-skim"
    assert ev_skim["cross_type"] == "skim"

    # --- EDR ---------------------------------------------------------
    ev_edr = {
        "drops": {"H+": 0.05, "e-": 0.75, "He+": 0.0, "O+": 0.0}
    }
    cat_edr = classify(ev_edr)
    assert cat_edr == "EDR"

    # --- plume -------------------------------------------------------
    ev_plume = {
        "drops": {"H+": 0.10, "e-": 0.15, "He+": 0.6, "O+": 0.55}
    }
    cat_plume = classify(ev_plume)
    assert cat_plume == "plume"

    # --- unknown -----------------------------------------------------
    ev_unknown = {
        "drops": {"H+": 0.3, "e-": 0.3, "He+": 0.2, "O+": 0.2}
    }
    cat_unk = classify(ev_unknown)
    assert cat_unk == "unknown"


# ---------------------------------------------------------------------
# find_candidates edge cases
# ---------------------------------------------------------------------
def test_find_candidates_handles_short_series(monkeypatch) -> None:
    """Detector should gracefully skip series shorter than ±pts window."""

    t = np.arange(0.0, 40.0, config.CADENCE_SEC, dtype=float)  # < 2 * pts_lead window
    entry = {
        "time_vi": t,
        "Vi": np.zeros((t.size, 3), dtype=float),
    }

    B = np.tile(np.array([[1.0, 0.0, 0.0]]), (t.size, 1))

    def fake_get_data(name: str):
        if name.endswith("fgm_b_gse_srvy_l2"):
            return t, B
        return t, np.ones_like(t)

    monkeypatch.setattr("mms_boundary_analysis.detect.flip_detector.get_data", fake_get_data)

    out = find_candidates({"mms1": entry})
    assert out["mms1"] == []
