#!/usr/bin/env bash
# Fetch model weights into the Hugging Face cache (~/.cache/huggingface), never into the repo.
# Needed only for the video and photo tiers and semantic labels; the LiDAR tier needs no weights. ~2.9 GB.
#   depth-anything/DA3-BASE                     Apache-2.0  multi-view / pose-conditioned depth
#   depth-anything/DA3METRIC-LARGE              Apache-2.0  metric scale
#   tue-mps/ade20k_semantic_eomt_large_512      MIT         indoor semantic segmentation (ADE20K)
# MapAnything (the old video/photo path, ROOMSCOPE_RECON=mapanything, 4.9 GB) only with --mapanything.
set -euo pipefail
PY="${PYTHON:-.venv/bin/python}"
REPOS="depth-anything/DA3-BASE depth-anything/DA3METRIC-LARGE tue-mps/ade20k_semantic_eomt_large_512"
[[ "${1:-}" == "--mapanything" ]] && REPOS="$REPOS facebook/map-anything-apache"
"$PY" - $REPOS <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    # safetensors + configs only: some repos also ship the same weights as pytorch_model.bin (EoMT: 2 x 1.26 GB)
    print(f"{repo} -> {snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.md'])}")
PYEOF
