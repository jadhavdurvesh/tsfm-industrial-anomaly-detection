"""
run_all_fault_types.py

Runs BOTH IsolationForest and Chronos across EVERY fault type discovered in
the attached dataset (inner race, outer race, ball — whatever is present),
instead of just one. Reuses the existing run_isolation_forest_baseline() and
run_chronos_zeroshot() functions unchanged — just calls each once per fault
type found, then prints + saves a comparison table.

Usage (same as the other scripts — run from inside src/):
    python run_all_fault_types.py
"""

import copy
import json
from datetime import datetime, timezone

from discover_dataset import discover_all_fault_pairs
from run_experiment import CONFIG as ISOFOREST_CONFIG, run_isolation_forest_baseline
from run_chronos_experiment import CONFIG as CHRONOS_CONFIG, run_chronos_zeroshot


def run_all():
    # Discover every fault type present in the attached dataset
    fault_pairs = discover_all_fault_pairs()

    if not fault_pairs:
        print("[run_all_fault_types] No fault files discovered under /kaggle/input — "
              "nothing to loop over. Attach a dataset first, or run the individual "
              "scripts directly to test in synthetic mode.")
        return

    summary = []

    for pair in fault_pairs:
        fault_type = pair["fault_type"]
        fault_size = pair["fault_size_inch"]
        tag = f"{fault_type}_{fault_size}"
        print(f"\n{'='*70}\nRunning all models for fault_type={fault_type}, size={fault_size}\n{'='*70}")

        # ---- IsolationForest -> logs/isolationforest/ ----
        iso_config = copy.deepcopy(ISOFOREST_CONFIG)
        iso_config["run_name"] = f"isolationforest_{tag}"
        iso_config["log_dir"] = "logs/isolationforest"
        iso_config["dataset"]["fault_type"] = fault_type
        iso_config["dataset"]["fault_size_inch"] = fault_size
        iso_results = run_isolation_forest_baseline(iso_config)

        # ---- Chronos zero-shot ("frozen") -> logs/chronos_zero_shot/ ----
        chronos_zs_config = copy.deepcopy(CHRONOS_CONFIG)
        chronos_zs_config["run_name"] = f"chronos_zeroshot_{tag}"
        chronos_zs_config["log_dir"] = "logs/chronos_zero_shot"
        chronos_zs_config["dataset"]["fault_type"] = fault_type
        chronos_zs_config["dataset"]["fault_size_inch"] = fault_size
        chronos_zs_config["finetuning"]["enabled"] = False
        chronos_zs_results = run_chronos_zeroshot(chronos_zs_config)

        # ---- Chronos fine-tuned -> logs/chronos_fine_tuned/ ----
        chronos_ft_config = copy.deepcopy(CHRONOS_CONFIG)
        chronos_ft_config["run_name"] = f"chronos_finetuned_{tag}"
        chronos_ft_config["log_dir"] = "logs/chronos_fine_tuned"
        chronos_ft_config["dataset"]["fault_type"] = fault_type
        chronos_ft_config["dataset"]["fault_size_inch"] = fault_size
        chronos_ft_config["finetuning"]["enabled"] = True
        chronos_ft_config["finetuning"]["learning_rate"] = 1e-4
        chronos_ft_config["finetuning"]["epochs"] = 3
        chronos_ft_config["finetuning"]["batch_size"] = 8
        chronos_ft_results = run_chronos_zeroshot(chronos_ft_config)

        summary.append({
            "fault_type": fault_type,
            "fault_size_inch": fault_size,
            "isolationforest": iso_results,
            "chronos_zero_shot": chronos_zs_results,
            "chronos_fine_tuned": chronos_ft_results,
        })

    # ---- Print comparison table ----
    print(f"\n{'='*70}\nSUMMARY — all fault types\n{'='*70}")
    header = f"{'Fault Type':<15}{'Size':<8}{'Model':<20}{'F1':<8}{'Precision':<12}{'Recall':<8}"
    print(header)
    print("-" * len(header))
    for row in summary:
        for model_name, key in [("IsolationForest", "isolationforest"),
                                 ("Chronos (zero-shot)", "chronos_zero_shot"),
                                 ("Chronos (fine-tuned)", "chronos_fine_tuned")]:
            results = row[key]
            print(f"{row['fault_type']:<15}{str(row['fault_size_inch']):<8}{model_name:<20}"
                  f"{results['f1_score']:<8.4f}{results['precision']:<12.4f}{results['recall']:<8.4f}")

    # ---- Save summary JSON ----
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = f"logs/summary_all_fault_types_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    run_all()
