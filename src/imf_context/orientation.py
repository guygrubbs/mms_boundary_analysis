"""
mms_boundary_analysis.imf_context.orientation
=============================================

Compute the **clock** and **cone** angles of the Interplanetary Magnetic
Field (IMF) together with a handful of handy solar-wind parameters
(dynamic pressure and Alfvén Mach number when the needed inputs are
present).

Definitions
-----------

* **Clock angle**  (deg)  
  :math:`\\theta_c = \\tan^{-1} \\bigl( B_Y / B_Z \\bigr)`, folded to
  *0 → 180°*.

* **Cone angle**   (deg)  
  :math:`\\theta_{cone} = \\cos^{-1} \\bigl( B_X / \\lVert\\mathbf B\\rVert \\bigr)`.

Input flexibility
-----------------

``get_angles`` accepts **either**

1. A single *dict‐like* object (typically the output of
   :pyfunc:`mms_boundary_analysis.io.omni.get_context`) **or**
2. Explicit keyword arguments ``Bx``, ``By``, ``Bz`` and optionally
   ``Vsw`` (km/s) and ``Nsw`` (cm⁻³).

All inputs are treated as **scalar floats**; vectorisation is a caller
responsibility.

Example
-------

```python
from mms_boundary_analysis.io import get_omni_ctx
from mms_boundary_analysis.imf_context import get_angles

ctx = get_omni_ctx('2019-01-27T12:15:00Z')
angles = get_angles(ctx)
print(angles['clock_deg'], angles['cone_deg'])
````

"""

from __future__ import annotations

from typing import Mapping, Dict

import numpy as np

# ---------------------------------------------------------------------

def _calc_angles(Bx: float, By: float, Bz: float) -> Dict[str, float]:
    """Return clock & cone angles plus |B| (all floats)."""
    Bmag = float(np.sqrt(Bx**2 + By**2 + Bz**2))
    if Bmag == 0.0:
        return dict(clock_deg=np.nan, cone_deg=np.nan, Bmag_nT=0.0)

    clock = float(np.degrees(np.arctan2(abs(By), Bz)))          # 0–180°
    cone  = float(np.degrees(np.arccos(Bx / Bmag)))             # 0–180°
    return dict(clock_deg=clock, cone_deg=cone, Bmag_nT=Bmag)


def _dynamic_pressure(N: float, V: float) -> float:
    """
    Dynamic pressure (nPa) from density (cm⁻³) and speed (km/s).

    Pd = 1.6726×10⁻⁶ · N · V²
    """
    return 1.6726e-6 * N * V * V

def _mach_alfven(Bmag: float, N: float, V: float) -> float:
    """
    Alfvén Mach number M\_A  (dimensionless).

    V_A (km/s) = 20.3 · B(nT) / sqrt(N cm⁻³)   →   M_A = V / V_A
    """
    if N <= 0.0 or Bmag == 0.0:
        return float("nan")
    v_a = 20.3 * Bmag / np.sqrt(N)
    return V / v_a

# ---------------------------------------------------------------------

def get_angles(
    ctx_or_Bx: "Mapping[str, float] | float",
    /,
    By: float | None = None,
    Bz: float | None = None,
    *,
    Vsw: float | None = None,
    Nsw: float | None = None,
) -> Dict[str, float]:
    """
    Compute IMF orientation metrics.

    ```
    Parameters
    ----------
    ctx_or_Bx : mapping | float
        Either a mapping with keys *Bx_nT, By_nT, Bz_nT* (plus
        optionally *Vsw_kms*, *Nsw_cm3*) **or** the scalar *Bx* value
        (nT) when supplying components explicitly.
    By, Bz : float, optional
        IMF components (nT) when *ctx_or_Bx* supplies *Bx* only.
    Vsw : float, optional
        Solar‐wind bulk speed (km/s).  Needed for *dynamic pressure*
        and *Alfvén Mach number*.
    Nsw : float, optional
        Proton density (cm⁻³).  Needed for *dynamic pressure* and
        *Alfvén Mach number*.

    Returns
    -------
    dict
        ``{ clock_deg, cone_deg, Bmag_nT, Pdyn_nPa?, M_A? }`` – the last
        two keys appear only if *Vsw* and *Nsw* were provided.
    """
    # Case 1: mapping passed
    if isinstance(ctx_or_Bx, Mapping):
        ctx = ctx_or_Bx
        Bx = float(ctx.get("Bx_nT"))
        By = float(ctx.get("By_nT"))
        Bz = float(ctx.get("Bz_nT"))
        Vsw = ctx.get("Vsw_kms", Vsw)
        Nsw = ctx.get("Nsw_cm3", Nsw)
    else:
        # explicit components
        Bx = float(ctx_or_Bx)
        if By is None or Bz is None:
            raise ValueError("By and Bz must accompany scalar Bx input")
        By = float(By)
        Bz = float(Bz)

    out = _calc_angles(Bx, By, Bz)

    if Vsw is not None and Nsw is not None:
        Pd = _dynamic_pressure(float(Nsw), float(Vsw))
        Ma = _mach_alfven(out["Bmag_nT"], float(Nsw), float(Vsw))
        out.update(Pdyn_nPa=Pd, M_A=Ma)

    return out


__all__ = ["get_angles"]

"""
### Features

* Accepts either a **context dict** (e.g. from `io.omni`) or explicit
  IMF components.
* Computes and returns:
  * `clock_deg` – 0 → 180 °
  * `cone_deg`  – 0 → 180 °
  * `Bmag_nT`
  * `Pdyn_nPa` and `M_A` **only** when *Vsw* **and** *Nsw* are supplied.
* All numeric work is handled by NumPy in a vector‐safe manner, but the
  function intends **scalar** inputs for clarity in the pipeline.
"""
