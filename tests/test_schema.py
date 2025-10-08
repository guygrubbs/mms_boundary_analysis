"""Schema consistency tests."""

from __future__ import annotations

from mms_boundary_analysis import config, csv_schema


def test_config_csv_columns_matches_schema() -> None:
    assert config.CSV_COLUMNS == list(csv_schema.COLUMNS)
