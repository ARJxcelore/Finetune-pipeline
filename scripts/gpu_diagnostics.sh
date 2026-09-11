#!/usr/bin/env bash
set -euo pipefail

echo '=== Host NVIDIA ==='
nvidia-smi

echo
echo '=== Docker version ==='
docker --version

echo
echo '=== Docker GPU passthrough ==='
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
