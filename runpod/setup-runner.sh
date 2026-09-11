#!/usr/bin/env bash
set -euo pipefail

echo "This script documents the final runner bootstrap steps."
echo "1) Create a Linux RunPod GPU Pod with Docker + NVIDIA support."
echo "2) In GitHub: Settings -> Actions -> Runners -> New self-hosted runner."
echo "3) Download the runner using the commands GitHub displays for your repository."
echo "4) Configure labels so the runner has: self-hosted, linux, x64, gpu."
echo "5) Start the runner service."
echo "6) Verify: docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi"
