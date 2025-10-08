"""Integration regression covering the published January 27 event."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from mms_boundary_analysis import events

TRANGE = ["2019-01-27/12:00:00", "2019-01-27/13:00:00"]


def _load_reference() -> dict:
    doc = Path(__file__).resolve().parents[1] / "docs" / "published_case_studies.md"
    match = re.search(r"```json\s*(\{.*?\})\s*```", doc.read_text(encoding="utf-8"), flags=re.DOTALL)
    if not match:
        raise RuntimeError("Published case study JSON missing")
    return json.loads(match.group(1))


def _skip_if_offline(exc: Exception) -> None:
    message = str(exc)
    if (
        "No internet" in message
        or "essential variables" in message
        or "No matching CDF" in message
    ):
        pytest.skip("MMS CDF download failed – offline environment")
    raise exc


def test_pipeline_matches_published_reference(tmp_path):
    reference = _load_reference()
    try:
        results = events.run_pipeline(trange=TRANGE, probes=["1"], quiet=True, output_dir=tmp_path)
    except RuntimeError as exc:
        _skip_if_offline(exc)

    assert "mms1" in results and results["mms1"], "Expected MMS1 events"
    event = results["mms1"][0]

    assert event["event_id"] == reference["event_id"]
    assert event["category"] == reference["category"]
    assert event["cross_type"] == reference["cross_type"]
    assert event["delta_N_model_km"] == pytest.approx(reference["delta_N_model_km"], rel=0.05)
    assert event["delta_N_local_ref_km"] == pytest.approx(reference["delta_N_local_ref_km"], rel=0.05)
    assert event["N_angle_ref_deg"] == pytest.approx(reference["N_angle_ref_deg"], abs=2.0)
    assert event["Bz_nT"] == pytest.approx(reference["Bz_nT"], abs=1.0)
    assert event["By_nT"] == pytest.approx(reference["By_nT"], abs=1.0)
    assert event["clock_deg"] == pytest.approx(reference["clock_deg"], abs=5.0)
    assert event["cone_deg"] == pytest.approx(reference["cone_deg"], abs=5.0)
    assert event["Vsw_kms"] == pytest.approx(reference["Vsw_kms"], abs=50.0)
    assert event["Pdyn_nPa"] == pytest.approx(reference["Pdyn_nPa"], abs=0.5)

    csv_path = tmp_path / "MMS1_events.csv.gz"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df.loc[0, "event_id"] == reference["event_id"]
    assert df.loc[0, "cross_type"] == reference["cross_type"]
    assert df.loc[0, "category"] == reference["category"]
    assert df.loc[0, "delta_N_model_km"] == pytest.approx(reference["delta_N_model_km"], rel=0.05)
