# MMS Boundary Analysis – Pre-update Task Plan

## Task 1 – Repair MMS data loader contract and coverage ✅

**Issue:** The pipeline assumes `mms_data` entries expose `time_vi`/`time_pos`, but `load_mms_data` currently returns `tvi`/`tpos`, so `find_candidates` and `events.run_pipeline` crash when indexing the dictionary.

The loader also clips data using fixed global start/stop times instead of the user-supplied `trange`, and HPCA downloads are requested with `notplot=True`, meaning heavy-ion densities referenced in `config.SPECIES` never land in pytplot for candidate scoring.

**Implementation:** Parse `trange` into UTC datetimes to drive the clipping window, rename or duplicate the exported keys to the names expected downstream, and ensure HPCA moments are available to `pytplot.get_data` (e.g. by setting `notplot=False` or caching the returned arrays).

**Testing:** Add a loader-focused unit test that monkeypatches `pyspedas`/`pytplot` to return synthetic arrays and asserts the contract (`time_vi`, `time_pos`, heavy-ion channels present). Exercise a minimal pipeline path under the same stubs to confirm no KeyErrors.

**Verification:** Manually inspect a short mocked run to confirm candidate detection receives the expected cadence arrays and that heavy-ion drops populate.

**Documentation:** Update docstrings/README references to the loader outputs and note any new optional arguments or behaviour.

**Cleanup:** Remove obsolete key names and temporary compatibility layers once all call-sites are updated; tidy any monkeypatch fixtures.

**Next steps:** With a reliable loader, subsequent physics refinements (e.g. improved thickness estimates) can rely on consistent inputs.

:::task-stub{title="Align loader outputs with detector/pipeline expectations"}
- [x] In `src/io/loader.py`, parse the incoming `trange` strings into timezone-aware datetimes and derive `t0`/`t1` from those values instead of the global defaults.
- [x] Update the returned dictionary to expose `time_pos`/`time_vi` keys (either by renaming the current entries or by providing both names during a deprecation window) so that `detect.flip_detector` and `events.run_pipeline` receive the expected arrays.
- [x] Change the HPCA download block to retain data in pytplot (e.g. `notplot=False`) or push the returned arrays into pytplot manually so that `get_data(f"{sid}_{var}")` works for the heavy-ion species listed in `config.SPECIES`.
- [x] Adjust any downstream references (`tests`, helper utilities) that still expect the old key names, and extend/introduce a unit test that mocks `pyspedas`/`pytplot` to confirm the loader returns the corrected structure and includes HPCA species.
- [x] Run `pytest -q` to ensure the new tests and existing suite pass, then refresh relevant documentation (module docstrings/README) to describe the contract.
:::

## Task 2 – Harden candidate detection windowing ✅

**Issue:** `find_candidates` indexes `k ± pts_lead` without guarding the bounds, so early/late samples either wrap around via negative indices or raise `IndexError`, breaking the scan and corrupting density drops.

**Implementation:** Restrict the loop to `pts_lead ≤ k < len(t_common) - pts_lead` (and bail out when the cadence series is too short), or equivalently guard inside the loop before accessing the offsets.

**Testing:** Add a synthetic test where the time series length is shorter than `2*pts_lead` and confirm no exception occurs, plus a regression that ensures edge samples are skipped rather than mis-referenced.

**Verification:** Manually inspect a run over mocked data to ensure the candidate list matches expectations near the start/end of the interval.

**Documentation:** Clarify in the docstring that detections require at least ±1 minute of data on either side.

**Cleanup:** Remove any defensive comments/TODOs that became obsolete after the fix.

**Next steps:** Re-run the physics validation notebooks once the detector is stable to confirm thresholds still behave.

:::task-stub{title="Guard flip_detector against out-of-range density lookups"}
- [x] In `src/detect/flip_detector.py`, adjust the candidate loop bounds (or add explicit checks) so that `k - pts_lead` and `k + pts_lead` are always within array limits before the densities are sampled.
- [x] If the shared cadence array is shorter than the required window, skip processing for that spacecraft with a clear log/debug message.
- [x] Extend `tests/test_detector.py` (or a new unit test) with mocked density arrays sized smaller than `2*pts_lead` to verify the function neither crashes nor wraps indices.
- [x] Run `pytest -q` to confirm the new tests and existing suite pass.
:::

## Task 3 – Compute physical boundary thickness for classification ✅

**Issue:** Events currently record `delta_N_local_km = 0` and call `classify_event` with `thickness_km=0.0`, so every crossing is flagged as “skim”, contradicting the intended cross/skim logic and misrepresenting the physics.

**Implementation:** After deriving the event normal, compute a signed distance profile around the crossing (e.g. via `delta_n_local` applied to MEC positions within the MVA window) to estimate |ΔN| across the layer, then pass that thickness into `classify_event`. Populate `delta_N_local_km`/`delta_N_local_ref_km` with meaningful values instead of placeholders.

**Testing:** Create a synthetic event with known geometry (positions along the normal) and assert the computed thickness exceeds 500 km and the cross_type is “cross”; likewise ensure a thin layer yields “skim”.

**Verification:** Plot sample events (once implemented) to confirm the ΔN time series and classifications align with expectations.

**Documentation:** Update README/docs to explain how thickness is estimated and the assumptions behind “cross” vs “skim”.

**Cleanup:** Remove temporary placeholder assignments and any unused imports once the full computation is in place.

**Next steps:** With honest thickness estimates, revisit publication figures/tables to incorporate the corrected classifications.

:::task-stub{title="Derive real ΔN thickness and feed cross_type classification"}
- [x] In `src/events.py`, build a local ΔN profile for each event (e.g. take MEC positions within the MVA window, apply `delta_n_local` relative to the crossing point, and measure the span across the sign change) and store both the instantaneous value and the inferred layer thickness.
- [x] Use the computed thickness when calling `classify_event`, and back-fill `delta_N_local_km`/`delta_N_local_ref_km` (and optionally `N_angle_ref_deg`) with the physically meaningful results instead of hard-coded zeros/NaNs.
- [x] Add/adjust tests (a new synthetic pipeline fixture is fine) that mock the necessary inputs to verify “cross” vs “skim” outcomes and the recorded ΔN values.
- [x] Document the thickness estimation method in the module docstring/README, then run `pytest -q` to ensure all tests, including the new ones, pass.
:::

## Task 4 – Restore CLI plotting imports ✅

**Issue:** `visual.timeseries` imports `COLUMNS` from `config`, but only `CSV_COLUMNS` exists; importing the module (and thus running `python -m mms_boundary_analysis.cli plot …`) raises an `AttributeError` before any plotting occurs.

**Implementation:** Remove the unused import or switch to `csv_schema.COLUMNS` if column metadata is genuinely required.

**Testing:** Add a lightweight test that imports `visual.timeseries` (or invokes the CLI `plot` subcommand under mocked CSV inputs) to ensure the module loads without error.

**Verification:** Manually run the CLI plotting entry point with sample CSV data to confirm plotting works.

**Documentation:** Update any references in docs that mention the import to avoid confusion.

**Cleanup:** Delete the stale import and any related dead code.

**Next steps:** After the import path is healthy, polish the quick-look plotting workflow for publication figures.

:::task-stub{title="Fix timeseries plotting import typo"}
- [x] Edit `src/visual/timeseries.py` to drop the nonexistent `from ..config import COLUMNS` import (or replace it with `from ..csv_schema import COLUMNS` if you actually need the list).
- [x] Add a regression test (e.g. in `tests/` or a new CLI-specific module) that simply imports `mms_boundary_analysis.visual.timeseries` or exercises the CLI `plot` command with mocked data to catch future import regressions.
- [x] Run `pytest -q` to ensure the new test passes alongside the existing suite.
:::

## Task 5 – Correct OMNI datetime handling ✅

**Issue:** Passing a `numpy.datetime64` to `get_context` fails because `pd.to_datetime(...).tz_convert('UTC')` expects a timezone-aware timestamp; this breaks the pipeline’s IMF lookups for numpy timestamps.

**Implementation:** Use `pd.to_datetime(..., utc=True)` or apply `.tz_localize('UTC')` before converting, ensuring numpy datetimes resolve cleanly.

**Testing:** Introduce a unit test that monkeypatches `_ensure_loaded()` to return a tiny DataFrame and asserts `get_context(np.datetime64(...))` succeeds.

**Verification:** Manually request context with both float seconds and numpy datetimes to confirm parity.

**Documentation:** Update the docstring to mention accepted datetime types now work as advertised.

**Cleanup:** Remove any redundant conversions once the fix is in place.

**Next steps:** With robust time handling, proceed to validate the OMNI cache against publication intervals.

:::task-stub{title="Handle numpy.datetime64 inputs in get_omni_ctx"}
- [x] Modify `src/io/omni.py` so that the numpy datetime branch calls `pd.to_datetime(t, utc=True)` (or `.tz_localize("UTC")`) instead of `.tz_convert("UTC")` on a naive timestamp.
- [x] Add a dedicated unit test (e.g. in `tests/`) that stubs `_ensure_loaded()` to return known data and asserts `get_context(np.datetime64("2019-01-27T12:00:00Z"))` returns the expected row.
- [x] Run `pytest -q` to confirm the new test and existing suite succeed.
:::

## Task 6 – Align CSV schema definitions ✅

**Issue:** `config.CSV_COLUMNS` omits `cross_type` while `csv_schema.COLUMNS` includes it, contradicting the comment that both share the same order and risking silent schema drift between writers and documentation.

**Implementation:** Consolidate on a single authoritative column list (e.g. import the schema into config or vice versa) so the code and docs use the same sequence.

**Testing:** Extend writer tests to assert both CSV and Parquet writers honour the unified schema, including `cross_type`.

**Verification:** Inspect a freshly written CSV to ensure column order matches expectations.

**Documentation:** Update inline comments and README references to point to the canonical schema definition.

**Cleanup:** Remove duplicate or conflicting constants after unification.

**Next steps:** With a consistent schema, regenerate sample outputs for publication appendices.

:::task-stub{title="Unify CSV column definitions across config and csv_schema"}
- [x] Decide on the single source of truth (e.g. keep `src/csv_schema.py::COLUMNS`) and update `src/config.py` to reference it instead of maintaining a divergent `CSV_COLUMNS` list.
- [x] Ensure the writers in `src/io/writer.py` validate against that unified list, then adjust any unit tests to cover the `cross_type` column explicitly.
- [x] Run `pytest -q` to verify the writer tests (and the rest of the suite) pass after the change.
- [x] Refresh documentation/comments that reference the column order so they point to the chosen source.
:::

## Task 7 – Exercise CLI entry points and quiet output ✅

**Issue:** The command-line interface lacks automated coverage, so regressions in argument parsing or plotting could slip through unnoticed.  Additionally, `save_csv` always prints a status line, breaking `--quiet` runs and polluting automation logs.

**Implementation:** Add pytest-driven smoke tests for the `run` and `plot` subcommands by mocking `run_pipeline`, `plot_timeseries`, and `matplotlib` to avoid heavy dependencies.  Update the writer helper to honour a caller-supplied `silent` flag so `events.run_pipeline` can suppress progress messages end-to-end.

**Testing:** Extend the suite with CLI-focused tests that assert correct argument propagation and figure handling, then rerun `pytest -q`.

**Verification:** Manually inspect captured call arguments during the tests to confirm the CLI builds the expected data dictionaries and bypasses `plt.show()` when saving figures.

**Documentation:** Record the new coverage in this plan so future contributors know the CLI is protected.

**Cleanup:** Remove any inline `print` statements guarded by the new quiet flag and keep the mock helper utilities local to the tests.

**Next steps:** With the CLI covered, evaluate adding golden-image comparisons for publication figures.

:::task-stub{title="Add CLI smoke tests and quiet writer support"}
- [x] Introduce pytest cases that exercise the `run` and `plot` commands with monkeypatched dependencies to avoid downloads and GUI calls.
- [x] Allow `save_csv` to run silently so pipeline callers can suppress console chatter.
- [x] Run `pytest -q` to ensure the expanded suite passes.
:::

## Task 8 – Stabilise Shue projection and IMF scatter plotting ✅

**Issue:** The secant-based solver underpinning the Shue model projection could lose its bracket and fall back to a radial approximation, while the IMF scatter quick-look produced unhelpful empty axes when no events were available.

**Implementation:** Strengthened the root finder with bracket swaps, bisection safeguards, and an adaptive search window before falling back, and taught the scatter helper to skip non-numeric metrics while displaying a friendly message when no data exist.

**Testing:** Added a regression that verifies the projected intersection actually lies on the Shue surface and expanded the Matplotlib tests to cover empty/NaN scatter cases.

**Verification:** Confirmed the updated tests exercise both the solver and plotting behaviours without triggering the radial fallback.

**Documentation:** Logged this audit work in the plan for traceability.

**Cleanup:** None required beyond the solver refactor.

**Next steps:** With the projection stable, future physics validation can focus on physics rather than numerical artefacts.

:::task-stub{title="Harden Shue projection solver and scatter plotting"}
- [x] Refine the normal-projection root finder to preserve brackets and expand the search radius before conceding to the radial distance.
- [x] Add regression coverage confirming the recovered intersection resides on the Shue surface within tolerance.
- [x] Update the IMF scatter helper to handle empty event collections and non-numeric metrics, backed by tests.
- [x] Run `pytest -q` to ensure the suite passes with the new coverage.
:::

---

### Testing

✅ Tests executed with `pytest -q`.
