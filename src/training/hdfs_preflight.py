"""Hardware/environment preflight check for the HDFS QLoRA experiment.

Run before training or evaluation to fail fast with a clear message instead
of a cryptic CUDA OOM, and to record the exact software/hardware stack for
reproducibility (see run_metadata.json in src.training.hdfs_pipeline).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Below this, 4-bit QLoRA of a 1.5B model + activations will very likely OOM.
MIN_VRAM_GIB = 6.0


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "cuda_available": False,
        "gpu_name": None,
        "gpu_vram_gib": None,
        "cuda_version": None,
        "torch_version": None,
        "transformers_version": None,
        "bitsandbytes_version": None,
        "peft_version": None,
        "axolotl_version": None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_vram_gib"] = round(props.total_memory / (1024**3), 2)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    for module_name, key in [
        ("transformers", "transformers_version"),
        ("bitsandbytes", "bitsandbytes_version"),
        ("peft", "peft_version"),
        ("axolotl", "axolotl_version"),
    ]:
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[key] = None

    return info


def check_environment(info: dict[str, Any], *, require_gpu: bool = True) -> list[str]:
    problems: list[str] = []
    if require_gpu and not info["cuda_available"]:
        problems.append("CUDA is not available; QLoRA training requires an NVIDIA GPU")
    if require_gpu and info["gpu_vram_gib"] is not None and info["gpu_vram_gib"] < MIN_VRAM_GIB:
        problems.append(
            f"Detected only {info['gpu_vram_gib']} GiB VRAM; this config expects >= {MIN_VRAM_GIB} GiB"
        )
    if info["bitsandbytes_version"] is None and require_gpu:
        problems.append("bitsandbytes is not installed; required for 4-bit QLoRA")
    return problems


def print_report(info: dict[str, Any], *, sequence_len: int, micro_batch: int, grad_accum: int, base_model: str) -> None:
    print("Preflight check")
    print("===============")
    print(f"CUDA available:        {info['cuda_available']}")
    print(f"GPU name:               {info['gpu_name']}")
    print(f"GPU VRAM:               {info['gpu_vram_gib']} GiB")
    print(f"CUDA version:           {info['cuda_version']}")
    print(f"PyTorch version:        {info['torch_version']}")
    print(f"Transformers version:   {info['transformers_version']}")
    print(f"BitsAndBytes version:   {info['bitsandbytes_version']}")
    print(f"PEFT version:           {info['peft_version']}")
    print(f"Axolotl version:        {info['axolotl_version']}")
    print()
    print(f"Base model:             {base_model}")
    print("Training mode:          QLoRA 4-bit")
    print(f"Sequence length:        {sequence_len}")
    print(f"Per-device batch:       {micro_batch}")
    print(f"Gradient accumulation:  {grad_accum}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--no-gpu-required", action="store_true", help="Allow CPU-only environments (docs/tests)")
    parser.add_argument("--output", default=None, help="Optional path to also write the report as JSON")
    args = parser.parse_args()

    info = collect_environment()
    print_report(
        info,
        sequence_len=args.sequence_len,
        micro_batch=args.micro_batch_size,
        grad_accum=args.grad_accumulation,
        base_model=args.base_model,
    )

    problems = check_environment(info, require_gpu=not args.no_gpu_required)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"environment": info, "problems": problems}, handle, indent=2)

    if problems:
        print("\nPREFLIGHT FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nPREFLIGHT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
