# syntax=docker/dockerfile:1
# One pinned image for preprocessing, training, inference and the demo notebook.
# CUDA runtime libraries come from the torch cu124 wheels; the host needs the NVIDIA driver and container toolkit.
FROM python:3.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp MPLCONFIGDIR=/tmp/mpl \
    SOLAR_ROOT=/work SOLAR_CACHE=/cache \
    TORCH_HOME=/cache/torch HF_HOME=/cache/hf
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /opt/requirements.txt
WORKDIR /work
