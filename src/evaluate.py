"""
evaluate.py

Shared evaluation metrics for anomaly detection, used identically by both
run_experiment.py (IsolationForest baseline) and run_chronos_experiment.py
(Chronos), so results are directly comparable — same metric code, same
point-adjustment rules, applied to both.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Standard point-adjustment protocol for time-series anomaly detection:
    if ANY point within a true anomaly segment is detected, mark the WHOLE
    segment as detected. This is common in the literature but inflates F1 —
    ALWAYS report whether you used this, since it changes numbers a lot.
    """
    adjusted = y_pred.copy()
    anomaly_state = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # backfill this segment
            j = i
            while j >= 0 and y_true[j] == 1:
                adjusted[j] = 1
                j -= 1
        elif y_true[i] == 0:
            anomaly_state = False
        if anomaly_state:
            adjusted[i] = 1
    return adjusted


def detection_delay(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """
    Samples between true fault onset (first y_true==1) and first correct
    detection (first y_pred==1 at or after onset). Returns -1 if never detected.
    """
    onset_indices = np.where(y_true == 1)[0]
    if len(onset_indices) == 0:
        return -1
    onset = onset_indices[0]
    detected_after_onset = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= onset))[0]
    if len(detected_after_onset) == 0:
        return -1
    return int(detected_after_onset[0] - onset)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, use_point_adjustment: bool = False) -> dict:
    """
    Single entry point used by both scripts. Returns a dict ready to pass
    straight into ExperimentLogger.log_results().
    """
    y_pred_eval = point_adjust(y_true, y_pred) if use_point_adjustment else y_pred

    return {
        "f1_score": float(f1_score(y_true, y_pred_eval, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_eval, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_eval, zero_division=0)),
        "detection_delay_samples": detection_delay(y_true, y_pred),
        "point_adjustment_used": use_point_adjustment,
    }


if __name__ == "__main__":
    # Smoke test with synthetic data
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    print(evaluate_predictions(y_true, y_pred, use_point_adjustment=False))
    print(evaluate_predictions(y_true, y_pred, use_point_adjustment=True))
