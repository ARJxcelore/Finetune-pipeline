"""Merge a trained LoRA adapter into its base model, producing a standalone model.

Standalone counterpart to the `axolotl merge-lora` step in hdfs_pipeline.py: it
needs only transformers + peft, so a run whose in-pipeline merge failed (that
step is deliberately non-fatal) can be merged afterwards on the host, without
the Axolotl image or the rendered train.yml.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def resolve_base_model(adapter_dir: Path, override: str | None) -> str:
    if override:
        return override
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise SystemExit(f"Not a LoRA adapter directory (no adapter_config.json): {adapter_dir}")
    base_model = json.loads(config_path.read_text(encoding="utf-8")).get("base_model_name_or_path")
    if not base_model:
        raise SystemExit(f"adapter_config.json has no base_model_name_or_path; pass --base-model: {config_path}")
    return str(base_model)


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def copy_tokenizer_files(adapter_dir: Path, output_dir: Path) -> list[str]:
    """Prefer the adapter's tokenizer files: training may have added pad/special tokens."""
    copied = []
    for name in TOKENIZER_FILES:
        source = adapter_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    return copied


def merge(adapter_dir: Path, output_dir: Path, base_model: str, device: str, dtype_name: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)

    # Load unquantized: merging LoRA deltas into 4-bit weights would either fail
    # or silently round the update away, so this dequantizes to `dtype` instead.
    print(f"Loading base model {base_model} ({dtype_name}) on {device} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

    print(f"Applying adapter {adapter_dir} ...", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir), dtype=dtype)

    print("Merging ...", flush=True)
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir} ...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    copied = copy_tokenizer_files(adapter_dir, output_dir)
    if not copied:
        print("Adapter dir has no tokenizer files; saving the base model tokenizer instead.")
        AutoTokenizer.from_pretrained(base_model, use_fast=True).save_pretrained(str(output_dir))


def write_metadata(output_dir: Path, adapter_dir: Path, base_model: str, device: str, dtype_name: str) -> None:
    import importlib.metadata as md

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except Exception:
            return None

    metadata = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter_dir": str(adapter_dir.resolve()),
        "dtype": dtype_name,
        "device": device,
        "merged_by": "src.training.merge_adapter",
        "package_versions": {name: version(name) for name in ("torch", "transformers", "peft", "safetensors")},
    }
    (output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model")
    parser.add_argument("--adapter", type=Path, required=True, help="Adapter dir containing adapter_config.json")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write the merged model into")
    parser.add_argument("--base-model", default=None, help="Defaults to adapter_config.json's base_model_name_or_path")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--force", action="store_true", help="Overwrite a non-empty --output directory")
    args = parser.parse_args()

    adapter_dir: Path = args.adapter
    output_dir: Path = args.output

    if not adapter_dir.is_dir():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"Output directory is not empty (use --force to overwrite): {output_dir}")

    base_model = resolve_base_model(adapter_dir, args.base_model)
    device = resolve_device(args.device)

    merge(adapter_dir, output_dir, base_model, device, args.dtype)
    write_metadata(output_dir, adapter_dir, base_model, device, args.dtype)

    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    print(f"MERGE SUCCEEDED: {output_dir} ({total_bytes / 1e9:.2f} GB)")
    print(f"Evaluate/serve it with: --model-path {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
