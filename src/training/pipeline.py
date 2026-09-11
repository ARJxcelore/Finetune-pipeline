from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=ROOT, env=merged_env, check=True)


def main() -> int:
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "artifacts" / run_id
    processed_dir = run_dir / "processed"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Validate raw dataset.
    run(["python", "-m", "src.data.validate", "data/raw/dataset.jsonl"])

    # 2) Deterministic split + fingerprint.
    run([
        "python", "-m", "src.data.prepare",
        "--input", "data/raw/dataset.jsonl",
        "--output-dir", str(processed_dir),
        "--valid-ratio", os.getenv("VALID_RATIO", "0.2"),
        "--seed", os.getenv("SEED", "42"),
    ])

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": os.getenv("BASE_MODEL"),
        "model_revision": os.getenv("MODEL_REVISION"),
        "git_sha": os.getenv("GITHUB_SHA"),
        "dataset": str(processed_dir / "dataset_manifest.json"),
        "hardware": os.getenv("GPU_INFO", "unknown"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 3) Render a run-specific Axolotl config.
    render_env = {
        **os.environ,
        "TRAIN_FILE": str(processed_dir / "train.jsonl"),
        "VALID_FILE": str(processed_dir / "valid.jsonl"),
        "OUTPUT_DIR": str(run_dir / "axolotl"),
        "PREPARED_DIR": str(run_dir / "prepared"),
        "WANDB_RUN_NAME": os.environ.get("WANDB_RUN_NAME") or f"{os.getenv('WANDB_PROJECT', 'fine-tune')}-{run_id}",
    }
    run(["python", "-m", "src.training.render_config", "--output", str(run_dir / "train.yml")], env=render_env)

    # 4) Train.
    run(["axolotl", "train", str(run_dir / "train.yml")])

    # 5) Locate best adapter. Axolotl writes final adapter into output_dir.
    adapter_dir = run_dir / "axolotl"
    # If a checkpoint exists, use the final root output; otherwise fail clearly.
    if not (adapter_dir / "adapter_config.json").exists():
        checkpoints = sorted(adapter_dir.glob("checkpoint-*"))
        if not checkpoints:
            raise SystemExit("No adapter found after training")
        adapter_dir = checkpoints[-1]

    # 6) Evaluate candidate.
    run([
        "python", "-m", "src.evaluation.evaluate",
        "--base-model", os.environ["BASE_MODEL"],
        "--adapter", str(adapter_dir),
        "--dataset", str(processed_dir / "valid.jsonl"),
        "--output", str(run_dir / "evaluation.json"),
        "--max-samples", os.getenv("EVAL_MAX_SAMPLES", "20"),
        "--max-new-tokens", os.getenv("EVAL_MAX_NEW_TOKENS", "128"),
        "--min-f1", os.getenv("EVAL_MIN_F1", "0.35"),
    ])

    # 7) Merge adapter into a standalone model. Dequantizes to bf16 for serving.
    # Axolotl's merge-lora writes the merged model to {output_dir}/merged (output_dir = run_dir/axolotl).
    merged_dir = run_dir / "axolotl" / "merged"
    merge_env = {**os.environ}
    run([
        "axolotl", "merge-lora", str(run_dir / "train.yml"),
        "--lora-model-dir", str(adapter_dir),
        "--dequant",
    ], env=merge_env)

    if not merged_dir.exists():
        raise SystemExit(f"Expected merged model directory not found: {merged_dir}")

    # 8) Upload model when configured.
    if os.environ.get("HF_MODEL_REPO"):
        upload_env = {
            **os.environ,
            "MODEL_DIR": str(merged_dir),
            "EVAL_FILE": str(run_dir / "evaluation.json"),
            "MANIFEST_FILE": str(run_dir / "run_manifest.json"),
            "RUN_ID": run_id,
        }
        run(["python", "-m", "src.registry.upload_to_hub"], env=upload_env)
    else:
        print("HF_MODEL_REPO not set; skipping Hub upload")

    print(f"PIPELINE SUCCEEDED: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
