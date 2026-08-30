"""
run_chronos_experiment.py

Fully functional Chronos zero-shot anomaly detection.

Same use_synthetic_data flag as run_experiment.py:
  True  -> runs fully offline right now to verify pipeline logic.
           Since chronos-forecasting needs internet+GPU to install/run, this
           mode uses a lightweight STUB forecaster (naive persistence: predict
           next value = last value) so you can test the windowing/scoring/
           logging pipeline without the real model. Clearly logged as STUB.
  False -> uses the REAL Chronos pipeline. Requires:
             pip install chronos-forecasting torch
           and a GPU (set model.device accordingly). Use this on Kaggle.
"""

import numpy as np

from experiment_logger import ExperimentLogger
from data_utils import (
    make_windows,
    window_labels,
    normalize_signal,
    load_cwru_mat_pair,
    generate_synthetic_bearing_signal,
)
from evaluate import evaluate_predictions
from discover_dataset import discover_cwru_dataset

# ============================================================
# 1. EXPLICIT CONFIG
# ============================================================
CONFIG = {
    "run_name": "chronos_zeroshot_cwru_innerrace_007",
    "seed": 42,

    "dataset": {
        "name": "CWRU",
        "use_synthetic_data": True,   # <-- set False on Kaggle with real Chronos + real data
        "source_url": "https://engineering.case.edu/bearingdatacenter",
        "fault_type": "inner_race",
        "fault_size_inch": 0.007,
        "load_hp": 1,
        "sampling_rate_hz": 12000,
        "sensor_position": "DE",
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
        "name": "chronos-t5-small",
        "mode": "zero-shot",
        "context_length": 1536,          # first N points of each window fed as context
        "prediction_length": 64,        # remaining points forecast and compared to actual
        "num_samples": 20,
        "anomaly_score_method": "forecast_mae",
        "device": "cuda",                # cuda | cpu
        "torch_dtype": "bfloat16",
    },

    "finetuning": {
        "enabled": False,
        "learning_rate": None,
        "epochs": None,
        "batch_size": None,
        "train_fraction_used": None,
    },

    "evaluation": {
        "metrics": ["f1", "precision", "recall", "detection_delay_samples"],
        "point_adjustment": False,
        "anomaly_threshold_method": "percentile_95",
    },
}


def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_dataset_config(dataset_cfg: dict) -> dict:
    """
    Same auto-discovery as run_experiment.py — checks /kaggle/input first,
    overrides use_synthetic_data + paths if a real pair is found, and prints
    exactly what changed. Falls back to synthetic mode silently otherwise.
    """
    discovered = discover_cwru_dataset(
        preferred_fault_type=dataset_cfg["fault_type"],
        preferred_fault_size_inch=dataset_cfg["fault_size_inch"],
        verbose=True,
    )
    if discovered is None:
        print("[resolve_dataset_config] No real dataset detected — using synthetic data "
              "(StubForecaster, plumbing test only).")
        return dataset_cfg

    resolved = dict(dataset_cfg)
    resolved["use_synthetic_data"] = False
    resolved["normal_file_path"] = discovered["normal_file_path"]
    resolved["fault_file_path"] = discovered["fault_file_path"]
    resolved["fault_type"] = discovered["fault_type"]
    resolved["fault_size_inch"] = discovered["fault_size_inch"]
    print(f"[resolve_dataset_config] Real dataset detected — switching to REAL Chronos + REAL DATA mode "
          f"(fault_type={discovered['fault_type']}, fault_size_inch={discovered['fault_size_inch']}).")
    return resolved


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


class StubForecaster:
    """
    Naive-persistence stand-in for Chronos, used ONLY when use_synthetic_data
    is True, so the pipeline can be tested without installing chronos-forecasting
    or using a GPU. Predicts that future values equal the last observed value —
    this is NOT a real baseline for your paper, it is a plumbing test only.
    """
    def predict(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        last_value = context[-1]
        return np.full(prediction_length, last_value)


class ChronosForecaster:
    """Real Chronos wrapper. Requires: pip install chronos-forecasting torch"""
    def __init__(self, model_name: str, device: str, torch_dtype: str):
        import torch
        from chronos import ChronosPipeline

        self.pipeline = ChronosPipeline.from_pretrained(
            f"amazon/{model_name}",
            device_map=device,
            torch_dtype=getattr(torch, torch_dtype),
        )

    def predict(self, context: np.ndarray, prediction_length: int, num_samples: int = 20) -> np.ndarray:
        import torch
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        try:
            forecast = self.pipeline.predict(
                context_tensor,  # positional — avoids relying on the exact param name
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
        except TypeError as e:
            # Fallback for versions where the keyword is named "inputs" instead of "context"
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
        # forecast shape: [num_series, num_samples, prediction_length] -> take median across samples
        median_forecast = np.median(forecast[0].numpy(), axis=0)
        return median_forecast


def load_forecaster(model_cfg: dict, use_synthetic_data: bool):
    if use_synthetic_data:
        return StubForecaster()
    return ChronosForecaster(
        model_name=model_cfg["name"],
        device=model_cfg["device"],
        torch_dtype=model_cfg["torch_dtype"],
    )


def compute_anomaly_scores(forecaster, windows: np.ndarray, context_length: int,
                            prediction_length: int, num_samples: int, use_synthetic_data: bool) -> np.ndarray:
    """
    For each window: use first `context_length` points as context, forecast
    the next `prediction_length` points, score = mean absolute error between
    forecast and actual continuation.

    NOTE: window_size must equal context_length + prediction_length for this
    to work — this is asserted in run_chronos_zeroshot() before calling this.
    """
    scores = np.zeros(len(windows))
    for i, window in enumerate(windows):
        context = window[:context_length]
        actual_future = window[context_length:context_length + prediction_length]

        if use_synthetic_data:
            forecast = forecaster.predict(context, prediction_length)
        else:
            forecast = forecaster.predict(context, prediction_length, num_samples)

        scores[i] = np.mean(np.abs(forecast - actual_future))
    return scores


def run_chronos_zeroshot(config: dict):
    # Resolve dataset config FIRST (may auto-switch synthetic -> real)
    config = dict(config)
    config["dataset"] = resolve_dataset_config(config["dataset"])

    logger = ExperimentLogger(run_name=config["run_name"])
    logger.log_seed(config["seed"])
    logger.log_config(config["dataset"])
    logger.log_config({"preprocessing": config["preprocessing"]})
    logger.log_model_params(config["model"])
    logger.log_training_params(config["finetuning"])
    logger.log_config({"evaluation": config["evaluation"]})

    use_synthetic = config["dataset"]["use_synthetic_data"]
    logger.log_note(
        "Chronos run — forecast-MAE zero-shot anomaly scoring. "
        f"Data source: {'SYNTHETIC (StubForecaster, plumbing test only)' if use_synthetic else 'REAL CWRU + REAL Chronos'}."
    )

    set_seed(config["seed"])

    # Sanity check: window_size must equal context_length + prediction_length
    window_size = config["preprocessing"]["window_size"]
    context_length = config["model"]["context_length"]
    prediction_length = config["model"]["prediction_length"]
    assert window_size == context_length + prediction_length, (
        f"window_size ({window_size}) must equal context_length + prediction_length "
        f"({context_length} + {prediction_length} = {context_length + prediction_length})"
    )

    # 1. Load data
    signal, fault_onset_idx = load_data(config["dataset"])
    logger.log_note(f"Signal length: {len(signal)}, fault onset index: {fault_onset_idx}")

    # 2. Preprocess
    signal_norm = normalize_signal(signal, config["preprocessing"]["normalization"])
    windows = make_windows(signal_norm, window_size, config["preprocessing"]["stride"])
    labels = window_labels(
        n_windows=len(windows),
        window_size=window_size,
        stride=config["preprocessing"]["stride"],
        fault_onset_idx=fault_onset_idx,
    )

    # 3. Train/test split (Chronos is zero-shot, so "train" split is unused for
    # fitting but kept for threshold calibration on "normal" data only)
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

    # 4. Load forecaster (stub if synthetic, real Chronos otherwise)
    forecaster = load_forecaster(config["model"], use_synthetic)

    # 5. Compute anomaly scores for train (to calibrate threshold) and test (to evaluate)
    train_scores = compute_anomaly_scores(
        forecaster, windows_train, context_length, prediction_length,
        config["model"]["num_samples"], use_synthetic,
    )
    test_scores = compute_anomaly_scores(
        forecaster, windows_test, context_length, prediction_length,
        config["model"]["num_samples"], use_synthetic,
    )

    # 6. Threshold: percentile of TRAIN scores only (assumes train is mostly normal)
    threshold_method = config["evaluation"]["anomaly_threshold_method"]
    assert threshold_method == "percentile_95", "Only percentile_95 implemented — extend here if you add others"
    threshold = np.percentile(train_scores, 95)
    preds = (test_scores > threshold).astype(int)
    logger.log_note(f"Anomaly threshold (95th percentile of train scores): {threshold:.6f}")

    # 7. Evaluate
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
    run_chronos_zeroshot(CONFIG)
