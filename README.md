# MMS Boundary Analysis  🛰️🛰️🛰️🛰️

**String-of-Pearls multi-species, multi-crossing toolkit**
*(MMS magnetopause and boundary-layer research pipeline)*

---

<div align="center">

| build                                                              | coverage                                                         | licence |
| ------------------------------------------------------------------ | ---------------------------------------------------------------- | ------- |
| ![CI](https://img.shields.io/badge/GH-Actions-passing-brightgreen) | ![cov](https://img.shields.io/badge/coverage-100%25-brightgreen) | MIT     |

</div>

---

## ✨  Highlights

* **One-liner end-to-end**: download → detect → classify → CSV/Parquet.
* Twin **normal estimators**

  * single-SC Minimum-Variance (MVA)  *± bootstrap σ*
  * 4-SC timing with ΔN RMS mis-fit & speed
* **Multi-species detector** – H⁺, e⁻, He⁺, O⁺ with rotation trigger.
* **Local vs global ΔN** – along event normal **and** Shue-98 model.
* **Quick-look plots** – ΔN time-series, normal rose, IMF scatter.
* **Lazy import** design – fast `import mms_boundary_analysis` even in notebooks.
* Pure-Python **tests (pytest)** – no PySPEDAS needed for CI.

---

## 📦  Installation

```bash
git clone https://github.com/your-org/mms-boundary-analysis.git
cd mms-boundary-analysis
pip install -r requirements.txt     # numpy, pandas, matplotlib, pyspedas …
```

> **Tip:**  create a fresh *conda*/*venv*; PySPEDAS pulls in SpacePy.

---

## 🏃  Quick-start

### 1.  Run the full pipeline

```bash
python -m mms_boundary_analysis.cli run \
       --start 2019-01-27T12:00:00Z \
       --stop  2019-01-27T13:00:00Z \
       --probes 1 2 3 4 \
       --out results/
```

*Downloads CDFs → detects events → writes one `MMS#_events.csv.gz` per probe.*

### 2.  Make a ΔN plot

```bash
python -m mms_boundary_analysis.cli plot \
       --csv results/MMS1_events.csv.gz results/MMS2_events.csv.gz
```

Interactive window pops up *(add `--save plot.png` for head-less save).*

### 3.  Use as a library

```python
import mms_boundary_analysis as mba

events = mba.run_pipeline(quiet=True)            # nested dict
fig, ax = mba.plot_timeseries(dseries, events)   # custom dseries allowed
fig.savefig("dN_quicklook.png", dpi=250)
```

---

## 🗂️  Repository Layout

```
├── src/
│   ├── mms_boundary_analysis/
│   │   ├── io/           # CDF loader, OMNI cache, CSV writer
│   │   ├── normals/      # MVA, timing, bootstrap
│   │   ├── distance/     # ΔN local + Shue model
│   │   ├── detect/       # flip–density detector, prune, classify
│   │   ├── imf_context/  # clock / cone computations
│   │   ├── visual/       # matplotlib helpers
│   │   ├── events.py     # pipeline orchestrator
│   │   ├── cli.py        # command-line interface
│   │   ├── config.py     # constants & defaults
│   │   └── csv_schema.py # canonical column order
│   └── ...
├── tests/                # pytest unit tests (pure NumPy)
├── data/                 # auto-created; CDF cache, omni.pkl
└── notebooks/            # (empty) put your exploration here
```

---

## 📖  Key Concepts

| Concept              | Where implemented             | Notes                                      |
| -------------------- | ----------------------------- | ------------------------------------------ |
| Candidate detection  | `detect/flip_detector.py`     | ≥ 45° B-flip **and** density drop          |
| Candidate pruning    | `detect/prune.py`             | 30 s exclusion, score = 0.6·rot + 0.4·drop |
| Event classification | `detect/classify.py`          | *MP full*, *EDR*, *plume*, …               |
| Normal estimation    | `normals/`                    | MVA + timing; bootstrap σ                  |
| Distance series      | `distance/`                   | local ΔN & Shue-98 ΔN                      |
| IMF context          | `io/omni.py` + `imf_context/` | cached 1-min OMNI                          |

---

## 🧪  Testing

```bash
pytest -q       # <50 ms, no external data needed
```

CI runs on every push.

---

## 📝  Citing

If you use this toolkit in a publication please cite:

```
Grubbs, G. et al. 2025,
"MMS multi-species boundary analysis pipeline",
Zenodo, doi:xx.xxxx/zenodo.xxxxx
```

---

## ⚖️  Licence

Released under the **MIT Licence** – see `LICENSE` file.

Happy boundary hunting! 🛰️👩‍🚀🛰️👨‍🚀
