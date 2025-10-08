# MMS Boundary Analysis 🛰️

**String-of-pearls multi-species, multi-crossing toolkit**
*(Magnetospheric Multiscale (MMS) magnetopause and boundary-layer pipeline)*

---

## ✨ Highlights

* **End-to-end automation** – download → detect → classify → export (CSV/Parquet).
* Dual **normal estimators** – single-spacecraft MVA with bootstrap σ and four-spacecraft timing.
* **Physics-grounded metrics** – ΔN along the event normal and the Shue-98 magnetopause.
* **Quick-look figures** – ΔN time series, normal roses, IMF scatter, FPI spectrograms, and FGM components.
* **Lazy imports** – heavy packages only load when the corresponding helper is used.

---

## 📦 Installation

```bash
git clone https://github.com/your-org/mms-boundary-analysis.git
cd mms-boundary-analysis
python -m venv .venv && source .venv/bin/activate  # optional but recommended
pip install -r requirements.txt
```

PySPEDAS will download SpacePy on first install; allow a few minutes for the compiled dependencies.

---

## 🏃 Quick-start

### 1. Run the full pipeline

```bash
python -m mms_boundary_analysis.cli run \
       --start 2019-01-27T12:00:00Z \
       --stop  2019-01-27T13:00:00Z \
       --probes 1 2 3 4 \
       --out results/
```

Downloads MMS CDFs (cached under `data/cdf/`), runs detection/classification, and writes one `MMS#_events.csv.gz` per probe.

### 2. Generate publication figures

```bash
python -m mms_boundary_analysis.cli plot \
       --csv results/MMS1_events.csv.gz --save dn_quicklook.png
```

For plasma/field context directly from MMS archives:

```python
from mms_boundary_analysis.visual.spectrograms import plot_fpi_spectrogram, plot_fgm_components

trange = ("2019-01-27/12:00:00", "2019-01-27/13:00:00")
plot_fpi_spectrogram("1", trange)
plot_fgm_components("1", trange)
```

### 3. Scripted use

```python
import pandas as pd

import mms_boundary_analysis as mba

events = mba.run_pipeline(
    trange=("2019-01-27/12:00:00", "2019-01-27/13:00:00"),
    probes=["1"],
    quiet=True,
)

df = pd.read_csv("results/MMS1_events.csv.gz")
dseries = {
    "mms1": {
        "time": pd.to_datetime(df["iso_time"]).to_numpy("datetime64[ns]"),
        "local": df["delta_N_local_km"].to_numpy(float),
        "model": df["delta_N_model_km"].to_numpy(float),
    }
}

fig, ax = mba.plot_timeseries(dseries, {"mms1": events["mms1"]})
fig.savefig("dn_quicklook.png", dpi=250)
```

---

## 🗂️ Repository layout

```
├── src/
│   ├── __init__.py          # public API & lazy imports
│   ├── cli.py               # command-line interface
│   ├── config.py            # constants & defaults
│   ├── csv_schema.py        # canonical column order
│   ├── detect/              # candidate detection & pruning
│   ├── distance/            # ΔN local + Shue magnetopause helpers
│   ├── events.py            # pipeline orchestrator
│   ├── imf_context/         # cone / clock angle helpers
│   ├── io/                  # MMS loader, OMNI cache, writers
│   ├── normals/             # MVA & timing normal estimators
│   └── visual/              # plotting utilities (timeseries, spectrograms, etc.)
├── tests/                   # pytest suite (network-enabled integration tests)
├── docs/                    # task plan, published references
└── requirements.txt         # runtime & test dependencies
```

Running the pipeline will create a `data/` directory that caches MMS CDFs and OMNI lookups.

---

## 📖 Key references

| Concept              | Implementation              | Notes                                                     |
| -------------------- | --------------------------- | --------------------------------------------------------- |
| Candidate detection  | `detect/flip_detector.py`   | ≥45° magnetic rotation and multi-species density drop     |
| Event classification | `detect/classify.py`        | Magnetopause classes with cross/skim type                 |
| Normal estimation    | `normals/`                  | Single-spacecraft MVA + four-spacecraft timing            |
| Distance series      | `distance/`                 | Local ΔN and Shue-98 model intersections                  |
| IMF context          | `io/omni.py` + `imf_context`| OMNI 1 min plasma/field context (cached locally)          |
| Publication plots    | `visual/`                   | ΔN quick-look, IMF scatter, FPI spectrograms, FGM traces  |

---

## 🧪 Testing & data integrity

```bash
pytest -q
```

The suite exercises the real MMS and OMNI downloads. If the NASA archives are unreachable the affected tests are skipped with a clear message; rerun the suite once connectivity is restored. The published January 27, 2019 case study is replayed end-to-end and compared against the values recorded in [`docs/published_case_studies.md`](docs/published_case_studies.md).

To refresh the cached inputs, remove the `data/` directory before running `pytest` or the CLI.

---

## 📝 Citing

If you use this toolkit in a publication please cite:

```
Grubbs, G. et al. (2025).
"MMS multi-species boundary analysis pipeline".
Zenodo. doi:xx.xxxx/zenodo.xxxxx
```

---

## ⚖️ Licence

Released under the **MIT License** – see [`LICENSE`](LICENSE).

Happy boundary hunting! 🛰️👩‍🚀🛰️👨‍🚀
