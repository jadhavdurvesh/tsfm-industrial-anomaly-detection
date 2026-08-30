"""
data_utils.py

Two ways to get data:
1. load_cwru_mat_pair()  — REAL data: loads two .mat files (a "normal" run and
   a "fault" run) and concatenates them into one signal with a known,
   documented fault-onset index. This is the standard way to turn CWRU's
   single-condition files into an anomaly-detection problem.
2. generate_synthetic_bearing_signal() — SYNTHETIC data: fabricates a signal
   with the same shape (normal region + injected fault region) so you can
   test the full pipeline (windowing, models, metrics, logging) end-to-end
   without downloading anything or using GPU time. Use this FIRST.

Both return: (signal: np.ndarray, fault_onset_idx: int)
"""

import numpy as np


def make_windows(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return np.array(windows)


def window_labels(n_windows: int, window_size: int, stride: int, fault_onset_idx: int) -> np.ndarray:
    """
    A window is labeled anomalous (1) if its midpoint falls at or after the
    fault onset index. Document this labeling rule explicitly in your paper —
    it's a common silent choice that affects reported metrics.
    """
    labels = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        window_start = i * stride
        window_mid = window_start + window_size // 2
        if window_mid >= fault_onset_idx:
            labels[i] = 1
    return labels


def normalize_signal(signal: np.ndarray, method: str) -> np.ndarray:
    if method == "z-score":
        return (signal - signal.mean()) / (signal.std() + 1e-8)
    elif method == "min-max":
        return (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    elif method == "none":
        return signal
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================
# REAL DATA: CWRU .mat file loading
# ============================================================
def load_cwru_mat_pair(normal_file_path: str, fault_file_path: str, sensor_position: str = "DE"):
    """
    Loads a CWRU "normal baseline" .mat file and a "fault" .mat file, extracts
    the vibration channel matching sensor_position (DE = drive end, FE = fan
    end), and concatenates normal -> fault into one signal.

    CWRU .mat files store the signal under a variable name like 'X097_DE_time'
    where the number is the file ID — it varies per file, so we search the
    keys for one matching the sensor_position suffix instead of hardcoding it.

    Returns: (concatenated_signal: np.ndarray, fault_onset_idx: int)
    """
    from scipy.io import loadmat

    def extract_channel(mat_path: str, suffix: str) -> np.ndarray:
        mat = loadmat(mat_path)
        matching_keys = [k for k in mat.keys() if k.endswith(f"{suffix}_time")]
        if not matching_keys:
            raise KeyError(
                f"No key ending in '{suffix}_time' found in {mat_path}. "
                f"Available keys: {[k for k in mat.keys() if not k.startswith('__')]}"
            )
        return mat[matching_keys[0]].flatten()

    normal_signal = extract_channel(normal_file_path, sensor_position)
    fault_signal = extract_channel(fault_file_path, sensor_position)

    fault_onset_idx = len(normal_signal)  # onset = exact concatenation point
    full_signal = np.concatenate([normal_signal, fault_signal])

    return full_signal, fault_onset_idx


# ============================================================
# SYNTHETIC DATA: for offline pipeline verification
# ============================================================
def generate_synthetic_bearing_signal(
    total_length: int = 24000,
    fault_onset_fraction: float = 0.6,
    sampling_rate_hz: int = 12000,
    base_freq_hz: float = 30.0,
    fault_freq_hz: float = 150.0,
    noise_std: float = 0.3,
    fault_amplitude: float = 1.5,
    seed: int = 42,
):
    """
    Fabricates a signal that mimics bearing vibration structure:
    - Normal region: clean low-frequency sinusoid (shaft rotation) + noise
    - Fault region: same base signal PLUS a higher-frequency, higher-amplitude
      component (mimicking impact events from a bearing defect)

    This is NOT real bearing physics — it exists purely so you can validate
    that your windowing/model/metric/logging pipeline runs correctly before
    touching real CWRU data or spending GPU time on Kaggle.

    Returns: (signal: np.ndarray, fault_onset_idx: int)
    """
    rng = np.random.RandomState(seed)
    t = np.arange(total_length) / sampling_rate_hz
    fault_onset_idx = int(total_length * fault_onset_fraction)

    base_signal = np.sin(2 * np.pi * base_freq_hz * t)
    noise = rng.normal(0, noise_std, size=total_length)
    signal = base_signal + noise

    fault_component = fault_amplitude * np.sin(2 * np.pi * fault_freq_hz * t)
    signal[fault_onset_idx:] += fault_component[fault_onset_idx:]

    return signal, fault_onset_idx


if __name__ == "__main__":
    # Smoke test: synthetic data -> windows -> labels
    signal, onset = generate_synthetic_bearing_signal()
    print(f"Signal length: {len(signal)}, fault onset at sample: {onset}")

    signal_norm = normalize_signal(signal, "z-score")
    windows = make_windows(signal_norm, window_size=2048, stride=512)
    labels = window_labels(len(windows), window_size=2048, stride=512, fault_onset_idx=onset)

    print(f"Num windows: {len(windows)}, num anomalous windows: {labels.sum()}")
