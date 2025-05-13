"""
mms_boundary_analysis.detect.flip_detector
==========================================

*First-pass* scan for candidate boundary events.

Strategy
--------
A sample index *k* is promoted to a **candidate** when **both**
conditions are satisfied:

1. **Magnetic-field rotation**  
   The 3-vector angle between **B(k-1)** and **B(k+1)** exceeds
   ``config.ROTATION_DEG`` (default ≈ 45°).

2. **Plasma density drop or rise**  
   For at least one species listed in ``config.SPECIES`` the fractional
   change across the same 2-sample window

       Δρ / ρ = |ρ(k-pts) − ρ(k+pts)| / ρ(k-pts)

   exceeds the species-specific threshold (e.g. 0.7 for H<sup>+</sup>,
   0.5 for O<sup>+</sup>, …).  ``pts`` is one minute worth of samples
   by default (15 indices at 4 s cadence).

Returned objects are *lightweight* dictionaries—heavy quantities
(time-series, large arrays) stay in memory inside other modules.

Public function
---------------

``find_candidates(spacecraft_data: dict)`` → ``dict[sc_id, list[cand]]``

where ``spacecraft_data`` is the dict produced by
:pyfunc:`mms_boundary_analysis.io.loader.load_mms_data`.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from pytplot import get_data

from .. import config


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _interp(src_t, src_y, tgt_t):
    """1-D linear interpolation with NaN for out-of-range targets."""
    if src_t is None:
        return np.full_like(tgt_t, np.nan, dtype=float)
    return np.interp(tgt_t, src_t, src_y, left=np.nan, right=np.nan)


def _rotation_angle(b_prev, b_next) -> float:
    """Angle (deg) between two 3-vectors."""
    num = np.dot(b_prev, b_next)
    den = np.linalg.norm(b_prev) * np.linalg.norm(b_next)
    if den == 0:
        return 0.0
    cosang = np.clip(num / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# ---------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------
def find_candidates(
    mms_data: Dict[str, dict],
    *,
    rotation_deg: float | None = None,
    drop_frac: float | None = None,
) -> Dict[str, List[dict]]:
    """
    Scan each spacecraft for boundary-crossing *candidates*.

    Parameters
    ----------
    mms_data : dict
        Output of :pyfunc:`mms_boundary_analysis.io.loader.load_mms_data`.
    rotation_deg : float or None
        Override for minimum magnetic rotation (deg).
    drop_frac : float or None
        Override for minimum density drop (as fraction, 0–1).  Species
        thresholds in ``config.SPECIES`` are **multiplied** by this
        factor.  ``None`` ⇒ use default (1.0).

    Returns
    -------
    dict
        ``{ sc_id: [ cand0, cand1, … ] }`` where each *cand* dict holds
        ``idx`` (sample index), ``time`` (POSIX s), ``rot_deg`` and
        ``drops`` mapping species→fraction.
    """
    rot_thr = rotation_deg or config.ROTATION_DEG
    drop_mult = 1.0 if drop_frac is None else float(drop_frac)
    pts_lead = int(60 / config.CADENCE_SEC)        # 1-min look-ahead

    out: Dict[str, List[dict]] = {}

    for sid, entry in mms_data.items():
        t_common = entry["time_vi"]                # 4-s cadence
        if t_common is None:
            continue

        # --- magnetic field (FGM) ------------------------------------------------
        tB, B = get_data(f"{sid}_fgm_b_gse_srvy_l2")
        if tB is None:
            continue
        B = B[:, :3] if B.shape[1] > 3 else B
        B_int = np.column_stack([_interp(tB, B[:, i], t_common) for i in range(3)])

        # --- species densities ---------------------------------------------------
        dens_int: Dict[str, np.ndarray] = {}
        for sp, (var, _) in config.SPECIES.items():
            tt, yy = get_data(f"{sid}_{var}")
            dens_int[sp] = _interp(tt, yy, t_common)

        # --- loop over interior samples -----------------------------------------
        candidates: List[dict] = []
        for k in range(1, len(t_common) - 1):
            # rotation angle
            rot = _rotation_angle(B_int[k - 1], B_int[k + 1])
            if rot < rot_thr:
                continue

            drops: Dict[str, float] = {}
            meets_drop = False
            for sp, (_, thresh) in config.SPECIES.items():
                arr = dens_int[sp]
                if np.isnan(arr[k - pts_lead]) or np.isnan(arr[k + pts_lead]):
                    frac = np.nan
                else:
                    pre = arr[k - pts_lead]
                    post = arr[k + pts_lead]
                    frac = abs(pre - post) / pre if pre > 0 else np.nan
                drops[sp] = frac

                if not np.isnan(frac) and frac >= thresh * drop_mult:
                    meets_drop = True

            if not meets_drop:
                continue

            candidates.append(
                dict(
                    idx=k,
                    time=float(t_common[k]),
                    rot_deg=rot,
                    drops=drops,
                )
            )

        out[sid] = candidates
    return out


__all__ = ["find_candidates"]
