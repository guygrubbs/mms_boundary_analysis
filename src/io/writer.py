"""
mms_boundary_analysis.io.writer
===============================

Utilities for persisting analysis products to disk—primarily **CSV**
(for portability) and **Parquet** (for efficiency).  All writers share
the same defaults:

*   Files are saved beneath :pydata:`mms_boundary_analysis.config.OUTPUT_DIR`.
*   Column order is validated against :pydata:`mms_boundary_analysis.config.CSV_COLUMNS`
    so downstream code sees a stable schema.
*   Parent directories are created on-the-fly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union, Sequence

import pandas as pd

from .. import config

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    """Raise ValueError if *df* is missing any expected column."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing expected columns: {', '.join(missing)}"
        )


def _ensure_parent(fn: Path) -> None:
    fn.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def save_csv(
    df: pd.DataFrame,
    filename: str | Path,
    *,
    include_index: bool = False,
    compress: bool = False,
) -> Path:
    """
    Save *df* to ``OUTPUT_DIR / filename`` as CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Table to write (must contain config.CSV_COLUMNS).
    filename : str | Path
        Relative filename (e.g. ``'MMS1_events.csv'``).  If an absolute
        path is supplied it is used verbatim.
    include_index : bool, default ``False``
        Whether to preserve the DataFrame index in the CSV.
    compress : bool, default ``False``
        If *True* appends ``'.gz'`` and writes gzip‐compressed CSV.

    Returns
    -------
    Path
        Full path of the written file.
    """
    _validate_columns(df, config.CSV_COLUMNS)

    out_path = Path(filename)
    if not out_path.is_absolute():
        out_path = config.OUTPUT_DIR / out_path
    if compress and not out_path.suffix.endswith(".gz"):
        out_path = out_path.with_suffix(out_path.suffix + ".gz")

    _ensure_parent(out_path)

    df.to_csv(out_path, index=include_index)
    print(f"[writer] CSV written → {out_path}")
    return out_path


def save_parquet(
    df: pd.DataFrame,
    filename: str | Path,
    *,
    compression: str = "snappy",
) -> Path:
    """
    Save *df* to Parquet under ``OUTPUT_DIR``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save (column check applied).
    filename : str | Path
        Relative or absolute path; ``.parquet`` is appended if missing.
    compression : {'snappy','gzip','brotli',None}
        Parquet compression codec (default *snappy*).

    Returns
    -------
    Path
        Path of the Parquet file on disk.
    """
    _validate_columns(df, config.CSV_COLUMNS)

    out_path = Path(filename)
    if not out_path.is_absolute():
        out_path = config.OUTPUT_DIR / out_path
    if out_path.suffix.lower() != ".parquet":
        out_path = out_path.with_suffix(".parquet")

    _ensure_parent(out_path)
    df.to_parquet(out_path, compression=compression)
    print(f"[writer] Parquet written → {out_path}")
    return out_path
