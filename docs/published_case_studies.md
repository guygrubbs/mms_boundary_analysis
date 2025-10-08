# Published Case Studies

## January 27, 2019 — 12:00–13:00 UTC

The Grubbs et al. (2025) publication reports the MMS1 magnetopause crossing in this interval with the following diagnostic values. These serve as a regression target for the automated pipeline.

```json
{
  "event_id": "MMS1_20190127T122816",
  "iso_time": "2019-01-27T12:28:16Z",
  "category": "MP full",
  "cross_type": "cross",
  "delta_N_local_km": 0.0,
  "delta_N_local_ref_km": 1040.0,
  "delta_N_model_km": 875.0,
  "N_angle_ref_deg": 11.5,
  "Bz_nT": 0.2,
  "By_nT": -2.1,
  "clock_deg": 135.0,
  "cone_deg": 22.0,
  "Vsw_kms": 420.0,
  "Pdyn_nPa": 1.8
}
```

The integration test in `tests/test_events.py` replays this interval with live MMS and OMNI downloads and asserts the pipeline output matches these figures within the quoted tolerances.
