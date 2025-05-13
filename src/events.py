"""
mms_boundary_analysis.events
============================

End-to-end **pipeline orchestrator**:

1.  Load MEC / FGM / FPI / HPCA data for the requested time range.
2.  Call :pyfunc:`detect.find_candidates`  → raw list of flips / drops.
3.  Call :pyfunc:`detect.prune_events`     → one “event” per boundary.
4.  For every surviving event
      a. Local MVA on a ±90 s window        → event normal **n̂_loc**
      b. Shue standoff + ΔN (local & model)
      c. IMF context via :pyfunc:`io.get_omni_ctx`
      d. Classification with :pyfunc:`detect.classify_event`
5.  Dump **per-spacecraft CSVs** that obey the canonical
   :pydata:`csv_schema.COLUMNS`.

The implementation is intentionally *lightweight*: no class hierarchy,
just functional steps that are easy to read and modify for a
case-study–driven workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config, csv_schema
from .io import load_mms_data, get_omni_ctx, save_csv
from .detect import find_candidates, prune_events, classify_event
from .normals import mva_normal, angle_between
from .distance import delta_n_local, delta_n_shue


# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------
MVA_WINDOW_SEC = 90        # ±90 s window around current-sheet index
MVA_WINDOW_SMP = int(MVA_WINDOW_SEC / config.CADENCE_SEC)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _mva_window(B: np.ndarray, k: int) -> np.ndarray:
    """Return B-matrix for MVA centred on sample *k* (handles edges)."""
    i0 = max(0, k - MVA_WINDOW_SMP)
    i1 = min(len(B), k + MVA_WINDOW_SMP)
    return B[i0:i1]


# ---------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------
def run_pipeline(
    trange: list[str] | None = None,
    probes: list[str] | None = None,
    *,
    output_dir: Path | None = None,
    quiet: bool = False,
) -> Dict[str, List[dict]]:
    """
    Execute full analysis and write one CSV per spacecraft.

    Parameters
    ----------
    trange : ['YYYY-mm-dd/HH:MM:SS', '…'] or None
        Time interval; defaults to ``config.DEFAULT_START / STOP``.
    probes : list[str] or None
        MMS probe numbers as strings, e.g. ``['1','2','3','4']``.
    output_dir : pathlib.Path or None
        Override CSV destination directory.
    quiet : bool
        If *True* suppresses console prints.

    Returns
    -------
    dict
        Nested events structure (mutated in place).
    """
    # ----- defaults ----------------------------------------------------
    trange  = trange or [
        config.DEFAULT_START.strftime("%Y-%m-%d/%H:%M:%S"),
        config.DEFAULT_STOP .strftime("%Y-%m-%d/%H:%M:%S"),
    ]
    probes  = probes or config.PROBES
    out_dir = output_dir or config.OUTPUT_DIR

    if not quiet:
        print(f"[events] Loading MMS data  {trange}  probes={','.join(probes)}")

    # ------------------------------------------------------------------
    mms_data = load_mms_data(trange)                         # step 1
    raw      = find_candidates(mms_data)                     # step 2
    events   = prune_events(raw)                             # step 3

    # ------------------------------------------------------------------
    for sc_id, ev_list in events.items():
        entry = mms_data[sc_id]
        t_common = entry["time_vi"]
        # interpolate B only once
        B_int = entry.get("_B_interp")
        if B_int is None:                                    # cache per SC
            from pytplot import get_data
            tB, B = get_data(f"{sc_id}_fgm_b_gse_srvy_l2")
            B = B[:, :3] if B.shape[1] > 3 else B
            B_int = np.column_stack([
                np.interp(t_common, tB, B[:, i], left=np.nan, right=np.nan)
                for i in range(3)
            ])
            entry["_B_interp"] = B_int

        # reference point: use position at candidate sample
        pos_t = entry["time_pos"]
        pos_r = entry["pos"]

        for ev in ev_list:
            k = ev["idx"]

            # ----- (a) local MVA --------------------------------------
            B_win = _mva_window(B_int, k)
            n_loc, lambdas, _ = mva_normal(B_win, return_full=True)
            ev["n_hat"] = n_loc
            ev["lambda_ratio"] = float(lambdas[1] / lambdas[2])

            # ----- (b) ΔN --------------------------------------------
            r0 = np.array([np.interp(t_common[k], pos_t, pos_r[:, i])
                           for i in range(3)])
            r_sc = r0                           # snapshot at cs time
            Pd  = ev["drops"].get("Pdyn_nPa")   # filled later from IMF
            # place-holder; actual Pd inserted below after IMF fetch

            ev["delta_N_local_km"] = 0.0        # trivially zero at cs
            ev["delta_N_model_km"] = np.nan     # placeholder

            # ----- (c) IMF context  ----------------------------------
            imf = get_omni_ctx(t_common[k])
            ev.update(imf)

            Pd = imf["Pdyn_nPa"]
            dN_model = delta_n_shue(r_sc, n_loc, Pd, imf["Bz_nT"])
            ev["delta_N_model_km"] = float(dN_model)

            # ----- (d) classification --------------------------------
            classify_event(ev, thickness_km=0.0)   # |ΔN|=0 at cs

            # event-level meta
            ev["event_id"] = config.make_event_id(sc_id.upper(), 
                                                  np.datetime64(int(t_common[k]), 's'))

    # ------------------------------------------------------------------
    # write per-probe CSVs
    for sc_id, ev_list in events.items():
        if not ev_list:
            continue
        rows = []
        for ev in ev_list:
            row = {c: np.nan for c in csv_schema.COLUMNS}
            row.update(
                iso_time = pd.to_datetime(ev["time"], unit="s", utc=True)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
                delta_N_local_km      = ev["delta_N_local_km"],
                delta_N_local_ref_km  = np.nan,           # filled later if needed
                delta_N_model_km      = ev["delta_N_model_km"],
                N_angle_ref_deg       = np.nan,
                Bz_nT     = ev["Bz_nT"],
                By_nT     = ev["By_nT"],
                clock_deg = ev["clock_deg"],
                cone_deg  = ev["cone_deg"],
                Vsw_kms   = ev["Vsw_kms"],
                Pdyn_nPa  = ev["Pdyn_nPa"],
                category  = ev["category"],
                cross_type = ev.get("cross_type", np.nan),
                event_id   = ev["event_id"],
            )
            rows.append(row)

        df = pd.DataFrame(rows, columns=csv_schema.COLUMNS)
        df = df.astype({k: csv_schema.DTYPES[k] for k in df.columns
                        if k in csv_schema.DTYPES})
        fname = f"{sc_id.upper()}_events.csv"
        save_csv(df, out_dir / fname, compress=True)

        if not quiet:
            print(f"[events] {sc_id}: wrote {len(df)} events → {fname}.gz")

    return events
