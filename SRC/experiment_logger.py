"""
experiment_logger.py

Purpose: guarantee that EVERY parameter used in an experiment run is captured
to disk, so results in the paper can always be traced back to an exact,
reproducible configuration. No silent defaults, no "I forgot what I set
learning_rate to."

Usage:
    from experiment_logger import ExperimentLogger

    logger = ExperimentLogger(run_name="chronos_zeroshot_cwru_0.007in")

    logger.log_config({
        "dataset": "CWRU",
        "fault_type": "inner_race",
        "fault_size_inch": 0.007,
        "sampling_rate_hz": 12000,
        "window_size": 2048,
        "stride": 512,
    })

    logger.log_model_params({
        "model_name": "chronos-t5-small",
        "mode": "zero-shot",
        "context_length": 512,
        "prediction_length": 64,
        "num_samples": 20,
    })

    logger.log_seed(42)

    # ... run your experiment ...

    logger.log_results({
        "f1_score": 0.83,
        "precision": 0.79,
        "recall": 0.87,
        "detection_delay_samples": 120,
    })

    logger.save()  # writes logs/<run_name>_<timestamp>.json
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


class ExperimentLogger:
    def __init__(self, run_name: str, log_dir: str = "logs"):
        self.run_name = run_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.record = {
            "run_name": run_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": self._capture_environment(),
            "dataset_config": {},
            "model_params": {},
            "training_params": {},
            "seed": None,
            "results": {},
            "notes": "",
        }

    # ---------- environment capture (automatic, no manual entry needed) ----------
    def _capture_environment(self) -> dict:
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        # Capture installed package versions relevant to reproducibility.
        # Extend this list as you add libraries (torch, transformers, chronos-forecasting, etc.)
        packages_to_check = [
            "numpy", "pandas", "scikit-learn", "torch", "transformers",
            "chronos-forecasting", "scipy", "matplotlib",
        ]
        versions = {}
        try:
            installed = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ).splitlines()
            installed_map = {}
            for line in installed:
                if "==" in line:
                    name, ver = line.split("==", 1)
                    installed_map[name.lower()] = ver
            for pkg in packages_to_check:
                versions[pkg] = installed_map.get(pkg.lower(), "not installed")
        except Exception as e:
            versions["error"] = f"could not capture pip freeze: {e}"
        env["package_versions"] = versions
        return env

    # ---------- explicit logging calls (you control what goes in) ----------
    def log_config(self, config: dict):
        """Dataset / preprocessing configuration."""
        self.record["dataset_config"].update(config)

    def log_model_params(self, params: dict):
        """Model architecture / inference configuration."""
        self.record["model_params"].update(params)

    def log_training_params(self, params: dict):
        """Fine-tuning / training hyperparameters (learning rate, epochs, batch size, etc.)."""
        self.record["training_params"].update(params)

    def log_seed(self, seed: int):
        self.record["seed"] = seed

    def log_results(self, results: dict):
        """Metrics: F1, precision, recall, detection delay, etc."""
        self.record["results"].update(results)

    def log_note(self, note: str):
        self.record["notes"] += (("\n" if self.record["notes"] else "") + note)

    def save(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{self.run_name}_{ts}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.record, f, indent=2, default=str)
        print(f"[ExperimentLogger] Saved full run record to {path}")
        return path


if __name__ == "__main__":
    # Smoke test
    logger = ExperimentLogger(run_name="smoke_test")
    logger.log_config({"dataset": "CWRU", "fault_size_inch": 0.007})
    logger.log_model_params({"model_name": "chronos-t5-small", "mode": "zero-shot"})
    logger.log_seed(42)
    logger.log_results({"f1_score": 0.0, "note": "placeholder run"})
    logger.save()
