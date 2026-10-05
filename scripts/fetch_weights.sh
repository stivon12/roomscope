#!/usr/bin/env bash
# Fetch model weights into the Hugging Face cache (~/.cache/huggingface), never into the repo.
# Needed for the video and photo tiers, semantic labels and damage; LiDAR geometry alone needs none. ~7.9 GB.
#   depth-anything/DA3-BASE                     Apache-2.0  multi-view / pose-conditioned depth
#   depth-anything/DA3METRIC-LARGE              Apache-2.0  metric scale
#   tue-mps/ade20k_semantic_eomt_large_512      MIT         indoor semantic segmentation (ADE20K)
#   IDEA-Research/grounding-dino-base           Apache-2.0  damage box proposals
#   google/siglip2-base-patch16-384             Apache-2.0  damage / clean-surface crop classifier
#   facebook/sam2.1-hiera-small                 Apache-2.0  damage masks
# MapAnything (the old video/photo path, ROOMSCOPE_RECON=mapanything, 4.9 GB) only with --mapanything.
set -euo pipefail
PY="${PYTHON:-.venv/bin/python}"
REPOS="depth-anything/DA3-BASE depth-anything/DA3METRIC-LARGE tue-mps/ade20k_semantic_eomt_large_512"
REPOS="$REPOS IDEA-Research/grounding-dino-base google/siglip2-base-patch16-384 facebook/sam2.1-hiera-small"
[[ "${1:-}" == "--mapanything" ]] && REPOS="$REPOS facebook/map-anything-apache"
"$PY" - $REPOS <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    # safetensors + configs + tokenizer files only: some repos also ship the same weights as pytorch_model.bin
    print(f"{repo} -> {snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.md', '*.txt', '*.model'])}")
PYEOF
