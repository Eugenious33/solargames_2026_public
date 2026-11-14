#!/usr/bin/env bash
# Build the image and record the fully resolved package set.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${SOLAR_IMAGE:-filament-seg:cu124}"
DOCKER_BUILDKIT=1 docker build -t "$IMAGE" "$HERE"
docker run --rm "$IMAGE" pip freeze > "$HERE/requirements.lock.txt"
docker run --rm --gpus all "$IMAGE" python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
