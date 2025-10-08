"""Quick-look spectrogram and magnetic-field plotting helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm

from pyspedas import mms
from pytplot import get_data

EPOCH_1970 = mdates.date2num(datetime(1970, 1, 1, tzinfo=timezone.utc))


def _to_mpl_seconds(time_arr: np.ndarray) -> np.ndarray:
    """Convert POSIX seconds or datetime64 arrays to Matplotlib numbers."""
    time_arr = np.asarray(time_arr)
    if np.issubdtype(time_arr.dtype, np.floating):
        return time_arr / 86400.0 + EPOCH_1970
    if np.issubdtype(time_arr.dtype, "datetime64"):
        sec = (time_arr - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        return sec.astype(float) / 86400.0 + EPOCH_1970
    raise TypeError("Unsupported time axis dtype for plotting")


def _ensure_fpi_energy(probe: str, trange: Iterable[str]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Ensure MMS FPI omni-directional energy spectrograms are in pytplot."""
    sid = f"mms{probe}"
    ion_name = f"{sid}_dis_energyspectr_omni_fast"
    ele_name = f"{sid}_des_energyspectr_omni_fast"
    ion_bins = f"{sid}_dis_energybins_fast"
    ele_bins = f"{sid}_des_energybins_fast"

    missing = [name for name in (ion_name, ele_name, ion_bins, ele_bins) if get_data(name) is None]
    if missing:
        mms.fpi(
            trange=list(trange),
            probe=probe,
            data_rate="fast",
            level="l2",
            datatype=["dis-dist", "des-dist"],
            notplot=False,
        )

    ion_spec = get_data(ion_name)
    ele_spec = get_data(ele_name)
    ion_energy = get_data(ion_bins)
    ele_energy = get_data(ele_bins)
    if None in (ion_spec, ele_spec, ion_energy, ele_energy):
        raise RuntimeError("FPI spectrogram data unavailable after download attempt")
    return (ion_spec, ion_energy), (ele_spec, ele_energy)


def plot_fpi_spectrogram(probe: str, trange: Iterable[str]) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Plot omni-directional ion and electron energy spectrograms."""
    (ion_spec, ion_energy), (ele_spec, ele_energy) = _ensure_fpi_energy(probe, trange)

    t_ion, spectr_ion = ion_spec
    _, energy_ion = ion_energy
    t_ele, spectr_ele = ele_spec
    _, energy_ele = ele_energy

    t_mpl_ion = _to_mpl_seconds(t_ion)
    t_mpl_ele = _to_mpl_seconds(t_ele)

    fig, (ax_ion, ax_ele) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))

    _plot_spectrogram(ax_ion, t_mpl_ion, energy_ion, spectr_ion, title=f"MMS{probe} Ion Spectrogram")
    _plot_spectrogram(ax_ele, t_mpl_ele, energy_ele, spectr_ele, title=f"MMS{probe} Electron Spectrogram")

    ax_ele.set_xlabel("UT")
    ax_ele.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, (ax_ion, ax_ele)


def _plot_spectrogram(ax: plt.Axes, t_mpl: np.ndarray, energy: np.ndarray, spectr: np.ndarray, *, title: str) -> None:
    """Helper for drawing a single omni-directional spectrogram panel."""
    energy = np.asarray(energy)
    if energy.ndim == 2:
        energy = np.nanmean(energy, axis=0)
    spectr = np.asarray(spectr)

    finite = spectr[np.isfinite(spectr)]
    positive = finite[finite > 0]
    vmin = float(np.nanmin(positive)) if positive.size else 1e-3
    vmax = float(np.nanmax(finite)) if finite.size else vmin * 10
    if vmax <= vmin:
        vmax = vmin * 10

    mesh = ax.pcolormesh(
        t_mpl,
        energy,
        spectr.T,
        shading="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
    )
    cbar = plt.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Differential Energy Flux")
    ax.set_yscale("log")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)


def plot_fgm_components(probe: str, trange: Iterable[str]) -> Tuple[plt.Figure, np.ndarray]:
    """Plot magnetic-field GSE components and magnitude for an MMS probe."""
    sid = f"mms{probe}"
    var_name = f"{sid}_fgm_b_gse_srvy_l2"
    if get_data(var_name) is None:
        mms.fgm(trange=list(trange), probe=probe, data_rate="srvy", level="l2", notplot=False)

    data = get_data(var_name)
    if data is None:
        raise RuntimeError("FGM magnetic-field data unavailable after download attempt")

    t, B = data
    B = np.asarray(B)
    if B.shape[1] >= 4:
        Bxyz = B[:, :3]
        Bmag = B[:, 3]
    else:
        Bxyz = B
        Bmag = np.linalg.norm(Bxyz, axis=1)

    t_mpl = _to_mpl_seconds(t)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 8))
    labels = ["Bx", "By", "Bz"]
    for i, label in enumerate(labels):
        axes[i].plot(t_mpl, Bxyz[:, i], lw=1.2)
        axes[i].set_ylabel(f"{label} (nT)")
        axes[i].grid(True, alpha=0.3)

    axes[3].plot(t_mpl, Bmag, lw=1.2, color="k")
    axes[3].set_ylabel("|B| (nT)")
    axes[3].set_xlabel("UT")
    axes[3].grid(True, alpha=0.3)

    axes[0].set_title(f"MMS{probe} Magnetic Field Components (GSE)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, axes


__all__ = ["plot_fpi_spectrogram", "plot_fgm_components"]
