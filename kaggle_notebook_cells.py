# ============================================================
# CELL 1 — Setup: clone repo using token-authenticated URL
# ============================================================
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")

GITHUB_USERNAME = "your-username"                     # <-- change this
REPO_NAME = "tsfm-industrial-anomaly-detection"        # <-- change this if you named it differently

repo_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

!git clone {repo_url}
%cd {REPO_NAME}

# Set git identity (required for commits from this environment)
!git config user.email "your-email@example.com"   # <-- change this
!git config user.name "your-name"                  # <-- change this


# ============================================================
# CELL 2 — Install dependencies
# ============================================================
!pip install -q -r requirements.txt


# ============================================================
# CELL 3 — Run ALL experiments (every fault type x every model variant)
# ============================================================
# This single script runs, for EVERY fault type discovered in your attached
# dataset (inner race, outer race, ball):
#   - IsolationForest              -> logs/isolationforest/
#   - Chronos zero-shot ("frozen") -> logs/chronos_zero_shot/
#   - Chronos fine-tuned           -> logs/chronos_fine_tuned/
# No path editing needed — auto-detects the dataset attached via Kaggle's
# "Add Data" panel. Prints a full comparison table at the end and saves a
# summary JSON. This is the ONE script you need — no separate runs required
# for a zero-shot vs. fine-tuned comparison, it's already all in one pass.
%cd src
!python run_all_fault_types.py
%cd ..


# ============================================================
# CELL 4 — Auto-commit and push logs back to GitHub
# ============================================================
import subprocess
from datetime import datetime, timezone

commit_message = f"Add run logs — {datetime.now(timezone.utc).isoformat()}"

!git add src/logs/
!git commit -m "{commit_message}"
!git push origin main

print("Logs pushed to GitHub.")
