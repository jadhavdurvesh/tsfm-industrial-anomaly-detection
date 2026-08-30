"""
discover_dataset.py

Scans /kaggle/input (Kaggle's standard mount point for attached datasets) for
.mat files, classifies each as "normal" or "fault" based on CWRU's standard
naming conventions, and reports exactly what it found — so you never have to
hardcode a file path again.

CWRU naming conventions this looks for (case-insensitive):
  - Normal baseline files: contain "normal" (e.g. "Normal_1.mat", "N_0.mat")
  - Fault files: contain a fault-type code:
      IR = inner race, OR = outer race, B = ball
    optionally followed by a size code (007, 014, 021, 028)
    e.g. "IR007_1.mat", "OR014@6_2.mat", "B021_3.mat"

If nothing is found (e.g. running locally with no /kaggle/input), this
returns None and the calling script falls back to synthetic data — it never
crashes, it just tells you clearly what happened.
"""

import os
import re
import glob


KAGGLE_INPUT_ROOT = "/kaggle/input"

FAULT_TYPE_PATTERNS = {
    # No trailing \b: CWRU filenames commonly use "_" right after the code
    # (e.g. "IR007_1.mat"), and "_" counts as a word character in regex, so
    # a trailing \b would fail to match right before it.
    "inner_race": re.compile(r"\bIR\d{3}", re.IGNORECASE),
    "outer_race": re.compile(r"\bOR\d{3}", re.IGNORECASE),
    "ball": re.compile(r"\bB\d{3}", re.IGNORECASE),
}
NORMAL_PATTERN = re.compile(r"normal", re.IGNORECASE)
FAULT_SIZE_PATTERN = re.compile(r"(007|014|021|028)")


def find_all_mat_files(root: str = KAGGLE_INPUT_ROOT):
    if not os.path.isdir(root):
        return []
    return glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)


def classify_file(filepath: str):
    """
    Returns a dict describing the file, or None if it doesn't match any
    known CWRU naming pattern.
    """
    filename = os.path.basename(filepath)

    if NORMAL_PATTERN.search(filename):
        return {"path": filepath, "category": "normal", "fault_type": None, "fault_size_inch": None}

    for fault_type, pattern in FAULT_TYPE_PATTERNS.items():
        if pattern.search(filename):
            size_match = FAULT_SIZE_PATTERN.search(filename)
            fault_size = f"0.{size_match.group(1)}" if size_match else None
            return {
                "path": filepath,
                "category": "fault",
                "fault_type": fault_type,
                "fault_size_inch": float(fault_size) if fault_size else None,
            }

    return None  # doesn't match any known pattern — ignored, not guessed at


def discover_cwru_dataset(preferred_fault_type: str = "inner_race",
                           preferred_fault_size_inch: float = 0.007,
                           root: str = None,
                           verbose: bool = True):
    """
    Scans, classifies, and picks ONE normal file + ONE fault file matching
    the preferred fault type/size (falls back to any fault file if no exact
    size match, so it still runs — but always reports exactly what it picked
    and why, so nothing is silently substituted without you knowing).

    Returns: dict with keys "normal_file_path", "fault_file_path",
             "fault_type", "fault_size_inch" — or None if nothing usable found.
    """
    if root is None:
        root = KAGGLE_INPUT_ROOT  # read fresh at call time, not bound at def time
    all_files = find_all_mat_files(root)

    if verbose:
        print(f"[discover_dataset] Scanning {root} ... found {len(all_files)} .mat file(s).")

    if not all_files:
        if verbose:
            print(f"[discover_dataset] No .mat files found under {root}. "
                  f"Falling back to synthetic data mode.")
        return None

    classified = [c for c in (classify_file(f) for f in all_files) if c is not None]
    unclassified = [f for f in all_files if classify_file(f) is None]

    normal_files = [c for c in classified if c["category"] == "normal"]
    fault_files = [c for c in classified if c["category"] == "fault"]

    if verbose:
        print(f"[discover_dataset] Classified: {len(normal_files)} normal file(s), "
              f"{len(fault_files)} fault file(s), {len(unclassified)} unrecognized file(s).")
        for f in normal_files:
            print(f"  NORMAL : {f['path']}")
        for f in fault_files:
            print(f"  FAULT  : {f['path']}  (type={f['fault_type']}, size={f['fault_size_inch']})")
        for f in unclassified:
            print(f"  IGNORED (unrecognized name pattern): {f}")

    if not normal_files or not fault_files:
        if verbose:
            print("[discover_dataset] Missing a normal file or a fault file — "
                  "cannot build a normal->fault pair. Falling back to synthetic data mode.")
        return None

    # Prefer an exact match on fault type + size; fall back to first available fault file
    exact_matches = [
        f for f in fault_files
        if f["fault_type"] == preferred_fault_type and f["fault_size_inch"] == preferred_fault_size_inch
    ]
    chosen_fault = exact_matches[0] if exact_matches else fault_files[0]
    chosen_normal = normal_files[0]

    if verbose:
        if exact_matches:
            print(f"[discover_dataset] Using exact match for {preferred_fault_type} "
                  f"{preferred_fault_size_inch}in: {chosen_fault['path']}")
        else:
            print(f"[discover_dataset] No exact match for {preferred_fault_type} "
                  f"{preferred_fault_size_inch}in — using first available fault file instead: "
                  f"{chosen_fault['path']} (type={chosen_fault['fault_type']}, size={chosen_fault['fault_size_inch']})")
        print(f"[discover_dataset] Using normal file: {chosen_normal['path']}")

    return {
        "normal_file_path": chosen_normal["path"],
        "fault_file_path": chosen_fault["path"],
        "fault_type": chosen_fault["fault_type"],
        "fault_size_inch": chosen_fault["fault_size_inch"],
    }


if __name__ == "__main__":
    result = discover_cwru_dataset()
    print("\nFinal result:", result)
