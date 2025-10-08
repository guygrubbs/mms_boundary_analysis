from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mms_boundary_analysis import cli, csv_schema


def _parse(args: list[str]):
    parser = cli._build_parser()
    return parser.parse_args(args)


def test_cli_run_invokes_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    called: dict[str, object] = {}

    def fake_run_pipeline(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    args = _parse(
        [
            "run",
            "--start",
            "2019-01-27T12:00:00Z",
            "--stop",
            "2019-01-27T12:05:00Z",
            "--probes",
            "1",
            "2",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    cli._cmd_run(args)

    assert called["trange"] == ["2019-01-27T12:00:00Z", "2019-01-27T12:05:00Z"]
    assert called["probes"] == ["1", "2"]
    assert called["output_dir"] == Path(tmp_path)
    assert called["quiet"] is True


class _DummyFig:
    def __init__(self, path_store: dict[str, Path]):
        self._store = path_store

    def tight_layout(self) -> None:
        self._store["tight"] = True

    def savefig(self, path: str | Path, dpi: int) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("dummy figure")
        self._store["saved_path"] = out
        self._store["dpi"] = dpi


def test_cli_plot_builds_timeseries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    row = {
        "iso_time": "2019-01-27T12:00:00Z",
        "delta_N_local_km": 10.0,
        "delta_N_local_ref_km": 600.0,
        "delta_N_model_km": 12.0,
        "N_angle_ref_deg": 5.0,
        "Bz_nT": 1.0,
        "By_nT": 0.5,
        "clock_deg": 120.0,
        "cone_deg": 30.0,
        "Vsw_kms": 400.0,
        "Pdyn_nPa": 2.0,
        "category": "mp_full",
        "cross_type": "cross",
        "event_id": "MMS1_20190127T120000",
    }
    df = pd.DataFrame([row], columns=csv_schema.COLUMNS)
    csv_path = tmp_path / "MMS1_events.csv"
    df.to_csv(csv_path, index=False)

    captured: dict[str, object] = {}

    def fake_plot_timeseries(dseries, events, **kwargs):
        captured["dseries"] = dseries
        captured["events"] = events
        captured["kwargs"] = kwargs
        return _DummyFig(captured), object()

    monkeypatch.setattr(cli, "plot_timeseries", fake_plot_timeseries)

    args = _parse(
        [
            "plot",
            str(csv_path),
            "--save",
            str(tmp_path / "plots" / "dn.png"),
        ]
    )

    cli._cmd_plot(args)

    assert "mms1" in captured["dseries"]
    dseries = captured["dseries"]["mms1"]
    assert set(dseries.keys()) == {"time", "local", "model"}
    assert captured["kwargs"]["title"] == "ΔN quick-look"
    assert captured["kwargs"]["show_model"] is True

    events_dict = captured["events"]
    assert events_dict["mms1"][0]["category"] == "mp_full"
    assert captured["saved_path"].name == "dn.png"


def test_cli_plot_no_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    row = {
        "iso_time": "2019-01-27T12:00:00Z",
        "delta_N_local_km": 5.0,
        "delta_N_local_ref_km": 300.0,
        "delta_N_model_km": 8.0,
        "N_angle_ref_deg": 10.0,
        "Bz_nT": 1.0,
        "By_nT": 0.5,
        "clock_deg": 90.0,
        "cone_deg": 45.0,
        "Vsw_kms": 350.0,
        "Pdyn_nPa": 1.5,
        "category": "mp_full",
        "cross_type": "skim",
        "event_id": "MMS2_20190127T120500",
    }
    df = pd.DataFrame([row], columns=csv_schema.COLUMNS)
    csv_path = tmp_path / "MMS2_events.csv"
    df.to_csv(csv_path, index=False)

    captured: dict[str, object] = {}

    def fake_plot_timeseries(dseries, events, **kwargs):
        captured["kwargs"] = kwargs
        return _DummyFig(captured), object()

    monkeypatch.setattr(cli, "plot_timeseries", fake_plot_timeseries)

    args = _parse(
        [
            "plot",
            str(csv_path),
            "--save",
            str(tmp_path / "plots" / "dn2.png"),
            "--no-model",
        ]
    )

    cli._cmd_plot(args)

    assert captured["kwargs"]["show_model"] is False

