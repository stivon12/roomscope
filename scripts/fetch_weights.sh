#!/usr/bin/env bash
# Fetch model weights into the Hugging Face cache (~/.cache/huggingface), never into the repo.
# MapAnything (Apache-2.0 variant, 1.23 B params, 4.9 GB): video and photo tiers.
set -euo pipefail
PY="${PYTHON:-.venv/bin/python}"
"$PY" - <<'PYEOF'
from huggingface_hub import snapshot_download
for repo in ["facebook/map-anything-apache"]:
    p = snapshot_download(repo)
    print(f"{repo} -> {p}")
PYEOF
