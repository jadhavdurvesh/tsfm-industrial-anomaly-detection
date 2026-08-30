# TSFM Industrial Anomaly Detection

Zero-shot and fine-tuned evaluation of time-series foundation models (Chronos)
against classical baselines (IsolationForest) for industrial bearing fault
detection on the CWRU dataset — written for an IAENG paper submission.

## Structure

```
src/
  experiment_logger.py       # auto-logs every parameter + environment info per run
  data_utils.py               # CWRU .mat loading + synthetic data generator for offline testing
  evaluate.py                  # shared F1/precision/recall/detection-delay metrics
  run_experiment.py            # IsolationForest baseline
  run_chronos_experiment.py    # Chronos zero-shot foundation model
  logs/                         # auto-generated JSON per run (full parameter + result record)
kaggle_notebook_cells.py       # cell-by-cell code to run this on Kaggle with GPU + auto git push
requirements.txt
```

## Quick start (offline, no GPU, no dataset needed)

Both experiment scripts default to `use_synthetic_data: True` in their
`CONFIG` dict, which generates a fabricated bearing-like signal so you can
verify the full pipeline (windowing → labeling → model → threshold →
metrics → logging) runs correctly before touching real data or GPU time.

```bash
cd src
pip install -r ../requirements.txt
python run_experiment.py            # IsolationForest baseline (real sklearn model)
python run_chronos_experiment.py    # Chronos pipeline (uses a naive-persistence
                                     # STUB forecaster in synthetic mode — NOT a
                                     # real result, just a plumbing test)
```

Each run writes a full JSON record to `src/logs/`.

## Running for real (Kaggle, GPU, real CWRU data)

See `kaggle_notebook_cells.py` — clones this repo, installs dependencies,
runs both experiments against real CWRU `.mat` files, and auto-commits the
resulting logs back to GitHub.

**You do not need to edit any file paths or config flags.** Both scripts
call `discover_dataset.discover_cwru_dataset()` automatically at the start
of every run, which:

1. Scans `/kaggle/input` recursively for `.mat` files
2. Classifies each as a "normal" baseline file or a "fault" file (and which
   fault type/size) based on CWRU's standard naming conventions
3. Prints exactly what it found and which files it picked
4. If a valid normal+fault pair is found, automatically overrides
   `use_synthetic_data` to `False` and wires in the real file paths
5. If nothing is found (e.g. no dataset attached yet), it falls back to
   synthetic mode and tells you clearly why

So on Kaggle: just attach any CWRU dataset via **Add Data**, run the script,
and check the printed `[discover_dataset]` / `[resolve_dataset_config]` lines
to confirm it found and used the right files. If it picks the wrong fault
type/size (e.g. no exact match for your preferred config), it tells you
exactly what it substituted and why — nothing is silently guessed.

In `run_chronos_experiment.py`, real-data mode also automatically switches
from `StubForecaster` to the real `ChronosForecaster` (requires
`pip install chronos-forecasting torch` and a GPU — both are in
`requirements.txt` and installed by the Kaggle notebook cells).

## Getting CWRU data

Search Kaggle Datasets for "CWRU bearing" for existing public mirrors, or
download directly from the
[Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter).

CWRU files are single-condition (all-normal or all-fault). This project turns
them into an anomaly-detection problem by concatenating a normal file and a
fault file, with the fault onset labeled at the exact concatenation point
(see `data_utils.load_cwru_mat_pair`).

## Reproducibility / audit trail

Every run — synthetic or real — writes a complete JSON to `src/logs/`
containing:
- Full dataset config (fault type, size, sensor position, sampling rate)
- Full preprocessing config (window size, stride, normalization)
- Full model config (architecture, hyperparameters, device, dtype)
- Random seed
- Python/library versions
- Final metrics

Commit these JSON files to the repo after every real run (the Kaggle
notebook does this automatically) so every number in the paper traces back
to an exact, versioned, reproducible configuration.
