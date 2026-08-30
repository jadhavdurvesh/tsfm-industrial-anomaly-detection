"""
run_experiment.py

Fully functional IsolationForest baseline. Runs TWO ways controlled by
CONFIG["dataset"]["use_synthetic_data"]:

  True  (default) -> uses generate_synthetic_bearing_signal(), runs fully
                      offline right now, no downloads needed. Use this to
                      verify the whole pipeline before touching real data.

  False -> loads real CWRU .mat files from CONFIG["dataset"]["normal_file_path"]
            and CONFIG["dataset"]["fault_file_path"]. Use this on Kaggle once
            you've attached the CWRU dataset.

Every parameter that affects the result lives in CONFIG below.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from experiment_logger import ExperimentLogger
from data_utils import (
    make_windows,
    window_labels,
    normalize_signal,
    load_cwru_mat_pair,
    generate_synthetic_bearing_signal,
)
from evaluate import evaluate_predictions

# ============================================================
# 1. EXPLICIT CONFIG
# ============================================================
CONFIG = {
    "run_name": "isolationforest_baseline_cwru_innerrace_007",
    "seed": 42,

    "dataset": {
        "name": "CWRU",
        "use_synthetic_data": True,   # <-- set False on Kaggle once real data is attached
        "source_url": "https://engineering.case.edu/bearingdatacenter",
        "fault_type": "inner_race",
        "fault_size_inch": 0.007,
        "load_hp": 1,
        "sampling_rate_hz": 12000,
        "sensor_position": "DE",
        # Only used when use_synthetic_data == False:
        "normal_file_path": "/kaggle/input/cwru-bearing-dataset/Normal_1.mat",
        "fault_file_path": "/kaggle/input/cwru-bearing-dataset/IR007_1.mat",
    },

    "preprocessing": {
        "window_size": 2048,
        "stride": 512,
        "normalization": "z-score",
        "filter": "none",
        "train_test_split": 0.7,
    },

    "model": {
        "name": "IsolationForest",
        "n_estimators": 100,
        "contamination": 0.05,
        "max_samples": "auto",
    },

    "evaluation": {
        "metrics": ["f1", "precision", "recall", "detection_delay_samples"],
        "point_adjustment": False,
    },
}


def set_seed(seed: int):
    np.random.seed(seed)


def load_data(dataset_cfg: dict):
    if dataset_cfg["use_synthetic_data"]:
        return generate_synthetic_bearing_signal(
            sampling_rate_hz=dataset_cfg["sampling_rate_hz"],
            seed=42,
        )
    else:
        return load_cwru_mat_pair(
            normal_file_path=dataset_cfg["normal_file_path"],
            fault_file_path=dataset_cfg["fault_file_path"],
            sensor_position=dataset_cfg["sensor_position"],
        )


def run_isolation_forest_baseline(config: dict):
    logger = ExperimentLogger(run_name=config["run_name"])
    logger.log_seed(config["seed"])
    logger.log_config(config["dataset"])
    logger.log_config({"preprocessing": config["preprocessing"]})
    logger.log_model_params(config["model"])
    logger.log_config({"evaluation": config["evaluation"]})
    logger.log_note(
        "Baseline run: IsolationForest on windowed vibration signal. "
        f"Data source: {'SYNTHETIC' if config['dataset']['use_synthetic_data'] else 'REAL CWRU'}."
    )

    set_seed(config["seed"])

    # 1. Load data
    signal, fault_onset_idx = load_data(config["dataset"])
    logger.log_note(f"Signal length: {len(signal)}, fault onset index: {fault_onset_idx}")

    # 2. Preprocess
    signal_norm = normalize_signal(signal, config["preprocessing"]["normalization"])
    windows = make_windows(
        signal_norm,
        window_size=config["preprocessing"]["window_size"],
        stride=config["preprocessing"]["stride"],
    )
    labels = window_labels(
        n_windows=len(windows),
        window_size=config["preprocessing"]["window_size"],
        stride=config["preprocessing"]["stride"],
        fault_onset_idx=fault_onset_idx,
    )

    # 3. Train/test split (chronological, not random shuffle — order matters for RUL/anomaly tasks)
    split_idx = int(len(windows) * config["preprocessing"]["train_test_split"])
    windows_train, windows_test = windows[:split_idx], windows[split_idx:]
    labels_test = labels[split_idx:]
    logger.log_config({
        "split_details": {
            "n_windows_total": len(windows),
            "n_windows_train": len(windows_train),
            "n_windows_test": len(windows_test),
            "n_anomalous_in_test": int(labels_test.sum()),
        }
    })

    # 4. Fit model (IsolationForest trains on train split, treating it as "mostly normal")
    model = IsolationForest(
        n_estimators=config["model"]["n_estimators"],
        contamination=config["model"]["contamination"],
        max_samples=config["model"]["max_samples"],
        random_state=config["seed"],
    )
    model.fit(windows_train)

    # 5. Predict on test split. sklearn's IsolationForest returns -1 for anomaly, 1 for normal.
    raw_preds = model.predict(windows_test)
    preds = (raw_preds == -1).astype(int)  # convert to 1 = anomaly, 0 = normal

    # 6. Evaluate
    results = evaluate_predictions(
        y_true=labels_test,
        y_pred=preds,
        use_point_adjustment=config["evaluation"]["point_adjustment"],
    )
    logger.log_results(results)
    logger.save()

    print("Results:", results)
    return results


if __name__ == "__main__":
    run_isolation_forest_baseline(CONFIG)
