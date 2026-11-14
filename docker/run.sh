#!/usr/bin/env bash
# Run a command of this repository inside the pinned GPU image, as the calling user.
#   docker/run.sh python src/fb/data.py test
#   docker/run.sh jupyter nbconvert --to notebook --execute --inplace demo.ipynb
# Mounts: repository -> /work, cache ($SOLAR_CACHE or ~/.cache/solar2026) -> /cache.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${SOLAR_CACHE:-$HOME/.cache/solar2026}"
mkdir -p "$CACHE"
exec docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" ${SOLAR_DOCKER_ARGS:-} \
  -v "$ROOT":/work -v "$CACHE":/cache -w /work "${SOLAR_IMAGE:-filament-seg:cu124}" "$@"
