"""
src/config.py
=============

Centralised configuration for the MMS boundary-analysis package.

Edit this file *only* when you want to change global defaults
(time-windows, thresholds, data directories, …).  All other modules
import from here, so a single place keeps the pipeline consistent.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Dict
import math
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Basic file-system layout
# ──────────────────────────────────────────────────────────────────────

# Project root (two parents above src/)
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Where raw CDFs and downloaded OMNI files live.
DATA_DIR: Path = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Output directory for per-spacecraft CSVs and quick-look plots
OUTPUT_DIR: Path = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Analysis window & spacecraft
# ──────────────────────────────────────────────────────────────────────

# Default event interval (UTC).  Can be overridden via CLI.
DEFAULT_START: datetime = datetime(2019, 1, 27, 12, 0, 0, tzinfo=timezone.utc)
DEFAULT_STOP:  datetime = datetime(2019, 1, 27, 13, 0, 0, tzinfo=timezone.utc)

# Probes analysed by default
PROBES: List[str] = ['1', '2', '3', '4']


# ──────────────────────────────────────────────────────────────────────
# Event-detection thresholds
# ──────────────────────────────────────────────────────────────────────

# Magnetic rotation threshold (deg) across boundary
ROTATION_DEG: float = 45.0

# Minimum fractional drop (or jump) in species density
DENSITY_DROP_FRAC: float = 0.5  # 50 %

# Sliding-window sizes
WINDOW_SEC:  int = 180   # MVA window ±180 s
CADENCE_SEC: int = 4     # analysis step used by detector
MIN_SEPARATION_SEC: int = 30  # do not count events closer than this

# Species table — mapping {label: (tplot-variable, drop_threshold)}
SPECIES: Dict[str, Tuple[str, float]] = {
    'H+':  ('dis_numberdensity_fast', 0.70),
    'e-':  ('des_numberdensity_fast', 0.70),
    'He+': ('hpca_heplus_numberdensity_fast', 0.50),
    'O+':  ('hpca_oplus_numberdensity_fast', 0.50),
}

# ──────────────────────────────────────────────────────────────────────
# Shue-98 model coefficients
#   r(θ) = r0 * (2 / (1 + cos θ))^alpha
#   r0(km) = a0 * P_dyn^a1 + a2 * Bz   (example parametrisation)
#   alpha  = b0 + b1 * Bz
#
# NOTE: The exact coefficients vary by publication.  The values below
#       match Shue et al. (1998) Table 2 for 5-min OMNI inputs.
#       Feel free to change if you prefer the 2001 update or Lin et al.
# ──────────────────────────────────────────────────────────────────────
SHUE_COEFFS = {
    "a0": 107.4,          # km · (nPa)^−a1
    "a1": -0.3333,        # exponent for P_dyn
    "a2": -0.28,          # km / nT
    "b0": 0.58,
    "b1": 0.007,
}

RE_KM: float = 6371.0  # Earth radius in kilometres

def shue_radius(theta_deg: float, P_dyn: float, Bz_nT: float) -> float:
    """
    Return magnetopause radial distance (kilometres) from Shue-98 model.

    Parameters
    ----------
    theta_deg : float
        Angle from subsolar point (deg).
    P_dyn : float
        Solar-wind dynamic pressure (nPa).
    Bz_nT : float
        IMF Bz in GSM (nT).

    Returns
    -------
    float
        Modelled magnetopause distance r(θ) in km.
    """
    a0, a1, a2, b0, b1 = (SHUE_COEFFS[k] for k in ("a0","a1","a2","b0","b1"))
    r0     = a0 * (P_dyn ** a1) + a2 * Bz_nT      # subsolar standoff (km)
    alpha  = b0 + b1 * Bz_nT
    theta  = math.radians(theta_deg)
    r_theta = r0 * ((2.0 / (1.0 + math.cos(theta))) ** alpha)
    return r_theta


# ──────────────────────────────────────────────────────────────────────
# Convenience: convert datetime → numpy datetime64 for fast math
# ──────────────────────────────────────────────────────────────────────
EPOCH64 = np.datetime64('1970-01-01T00:00:00Z')

def to_dt64(ts: datetime) -> np.datetime64:
    """UTC datetime → numpy.datetime64[ns]."""
    return EPOCH64 + np.int64(ts.timestamp() * 1e9)


# ──────────────────────────────────────────────────────────────────────
# Output column order (shared by writer & csv_schema)
# ──────────────────────────────────────────────────────────────────────
CSV_COLUMNS: List[str] = [
    'iso_time',
    'delta_N_local_km',      # (r-r0)·N̂  (event MVA normal)
    'delta_N_local_ref_km',  # same with reference normal
    'delta_N_model_km',      # radial Shue (or projected) distance
    'N_angle_ref_deg',       # angle between local & ref normal
    'Bz_nT', 'By_nT',
    'clock_deg', 'cone_deg',
    'Vsw_kms', 'Pdyn_nPa',
    'category',              # e.g. Full_MP, Skim_EDR…
    'event_id'
]

# ──────────────────────────────────────────────────────────────────────
# Helper for generating an event ID
# ──────────────────────────────────────────────────────────────────────
def make_event_id(sc: str, when: np.datetime64) -> str:
    """Return a unique event string 'MMS1_YYYYMMDDThhmmss'."""
    dt = str(when.astype('datetime64[s]'))
    return f"MMS{sc}_{dt.replace('-', '').replace(':', '')}"
