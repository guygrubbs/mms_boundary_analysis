"""
mms_boundary_analysis.cli
=========================

Minimal **command-line interface** that lets you run the entire pipeline
from a terminal prompt without touching Python.

Usage
-----

.. code-block:: console

    $ python -m mms_boundary_analysis.cli run \
          --start 2019-01-27T12:00:00Z \
          --stop  2019-01-27T13:00:00Z \
          --probes 1 2 3 4 \
          --out   results/

will download the required MMS CDFs, detect / classify crossings, and
write per-spacecraft ``*.csv.gz`` files under *results/*.

You can immediately make a quick-look plot:

.. code-block:: console

    $ python -m mms_boundary_analysis.cli plot \
          --csv results/MMS1_events.csv.gz results/MMS2_events.csv.gz

Dependencies
------------
Only the standard library + the project’s own modules are required.  No
external plotting is triggered unless you call *plot*.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from . import config
from .events import run_pipeline
from .visual import plot_timeseries


# ──────────────────────────────────────────────────────────────────────
# helper: ISO8601 → datetime
# ──────────────────────────────────────────────────────────────────────
def _iso(s: str) -> str:
    """
    Return ISO string with trailing 'Z' if no timezone was given.
    """
    if s.endswith("Z") or s[-6] in ["+", "-"]:  # simple check
        return s
    return s + "Z"


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(_iso(s).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


# ──────────────────────────────────────────────────────────────────────
# sub-command: run
# ──────────────────────────────────────────────────────────────────────
def _cmd_run(args: argparse.Namespace) -> None:
    tr = [args.start, args.stop]
    run_pipeline(
        trange=tr,
        probes=args.probes,
        output_dir=Path(args.out),
        quiet=args.quiet,
    )


# ──────────────────────────────────────────────────────────────────────
# sub-command: plot
# ──────────────────────────────────────────────────────────────────────
def _cmd_plot(args: argparse.Namespace) -> None:
    # Concatenate CSVs; assume they share the schema
    df_all = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)

    # Very compact ΔN plot: use local and model if present
    dser = {}
    for sc in sorted({c.split("_")[0].lower() for c in df_all["event_id"]}):
        sc_df = df_all[df_all["event_id"].str.startswith(sc.upper())]
        if sc_df.empty:
            continue
        t = pd.to_datetime(sc_df["iso_time"]).to_numpy("datetime64[ns]")
        dser[sc] = dict(
            time=t,
            local=sc_df["delta_N_local_km"].to_numpy(float),
            model=sc_df["delta_N_model_km"].to_numpy(float),
        )

    # Fake events dict for markers – only category and time needed
    evs = {sc: [] for sc in dser}
    for _, row in df_all.iterrows():
        sc = row["event_id"].split("_")[0].lower()
        evs[sc].append(
            dict(
                time=pd.to_datetime(row["iso_time"]).timestamp(),
                category=row["category"],
            )
        )

    fig, _ = plot_timeseries(
        dser,
        evs,
        title="ΔN quick-look",
        show_model=not args.no_model,
    )
    fig.tight_layout()
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=250)
        print(f"[cli] Plot saved → {args.save}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


# ──────────────────────────────────────────────────────────────────────
# CLI definition
# ──────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mms-boundary-analysis",
        description="Quick command-line wrapper around the MMS boundary pipeline",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ---------------------------------------------------------
    run_p = sub.add_parser("run", help="run full analysis")
    run_p.add_argument(
        "--start",
        type=_iso,
        default=config.DEFAULT_START.isoformat().replace("+00:00", "Z"),
        help="UTC ISO start time  (default pipeline default)",
    )
    run_p.add_argument(
        "--stop",
        type=_iso,
        default=config.DEFAULT_STOP.isoformat().replace("+00:00", "Z"),
        help="UTC ISO stop time   (default pipeline default)",
    )
    run_p.add_argument(
        "--probes",
        nargs="+",
        default=config.PROBES,
        help="list of MMS probes, e.g. 1 2 3 4",
    )
    run_p.add_argument(
        "--out",
        default=str(config.OUTPUT_DIR),
        help="output directory for CSVs",
    )
    run_p.add_argument(
        "--quiet", action="store_true", help="suppress progress messages"
    )
    run_p.set_defaults(func=_cmd_run)

    # ---- plot --------------------------------------------------------
    plot_p = sub.add_parser("plot", help="quick-look ΔN plot from CSVs")
    plot_p.add_argument("csv", nargs="+", help="per-spacecraft CSV.gz files")
    plot_p.add_argument("--no-model", action="store_true", help="hide ΔN model curve")
    plot_p.add_argument("--save", metavar="PNG", help="write figure instead of show()")
    plot_p.set_defaults(func=_cmd_plot)

    return p


# ──────────────────────────────────────────────────────────────────────
# entry-point
# ──────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:  # pragma: no cover
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


# ---------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
