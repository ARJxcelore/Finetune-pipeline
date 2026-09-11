"""Orchestrates the HDFS anomaly-classification experiment end to end.

    preflight (GPU/software check)
        v
    render HDFS Axolotl config
        v
    axolotl train
        v
    locate adapter
        v
    merge adapter into a standalone model (src.training.merge_adapter)
        v
    evaluate on the held-out test split (src.evaluation.hdfs_evaluate)
        v
    release gate (src.evaluation.hdfs_gate) -- fails closed
        v
    write run_metadata.json (seed, dataset source, base model, config, package
    versions, GPU, CUDA, git commit -- see docs/HDFS_ANOMALY_DETECTION.md #reproducibility)
        v
    upload to the HF model repo, ONLY behind a passing gate

Dataset build/split (src.data.hdfs.build_dataset) is still a separate step --
run `python -m src.cli dataset prepare --name hdfs` first. Base-vs-fine-tuned
comparison (src.evaluation.hdfs_compare) also stays separate, since it needs a
baseline run this pipeline does not produce.

Runs inside the pinned Axolotl image, same as src.training.pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.hdfs_gate import format_report, gate_report, thresholds_from_env

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = "configs/qwen2.5-1.5b-hdfs-qlora.yml"
PROCESSED_DIR = ROOT / "artifacts" / "hdfs" / "processed"

REQUIRED_ENV_DEFAULTS = {
    "BASE_MODEL": "Qwen/Qwen2.5-1.5B-Instruct",
    "MODEL_REVISION": "main",
    "SEQUENCE_LEN": "2048",
    "MICRO_BATCH_SIZE": "1",
    "GRAD_ACCUMULATION": "8",
    "EPOCHS": "2",
    "LEARNING_RATE": "0.0002",
    "SEED": "42",
    "USE_WANDB": "true",
    "WANDB_PROJECT": "hdfs-anomaly-classification",
    "WANDB_ENTITY": "",
}


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, cwd=ROOT, env=merged_env, check=True)


def collect_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name in ("torch", "transformers", "peft", "bitsandbytes", "axolotl", "datasets"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = None
    return versions


def write_train_subset(
    source: Path,
    dest: Path,
    n: int,
    *,
    anomaly_ratio: float | None,
    seed: int,
) -> dict[str, int]:
    """Write an n-example training subset, optionally rebalanced toward anomalies.

    Streams the source file rather than reading it whole -- train.jsonl is ~1.4 GB
    and this runs on a 16 GB machine.
    """
    import json as _json

    if anomaly_ratio is None:
        written = 0
        with source.open("r", encoding="utf-8") as src, dest.open("w", encoding="utf-8") as out:
            for line in src:
                if not line.strip():
                    continue
                out.write(line if line.endswith("\n") else line + "\n")
                written += 1
                if written >= n:
                    break
        print(f"Training subset: first {written} examples (natural class distribution) -> {dest}")
        return {"total": written}

    if not 0 < anomaly_ratio < 1:
        raise SystemExit("--train-anomaly-ratio must be between 0 and 1")

    want_anomaly = int(round(n * anomaly_ratio))
    want_normal = n - want_anomaly
    # Reservoir-free approach: collect only what we need, but read the whole file so
    # the picks aren't biased toward the head of the split.
    anomaly_lines: list[str] = []
    normal_lines: list[str] = []
    with source.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            label = _json.loads(line).get("label")
            (anomaly_lines if label == "anomaly" else normal_lines).append(line)

    rng = random.Random(seed)
    picked_anomaly = rng.sample(anomaly_lines, min(want_anomaly, len(anomaly_lines)))
    picked_normal = rng.sample(normal_lines, min(want_normal, len(normal_lines)))
    combined = picked_anomaly + picked_normal
    rng.shuffle(combined)

    with dest.open("w", encoding="utf-8") as out:
        for line in combined:
            out.write(line if line.endswith("\n") else line + "\n")

    counts = {"total": len(combined), "anomaly": len(picked_anomaly), "normal": len(picked_normal)}
    achieved = counts["anomaly"] / counts["total"] * 100 if counts["total"] else 0.0
    print(
        f"Training subset: {counts['total']} examples "
        f"({counts['anomaly']} anomaly / {counts['normal']} normal = {achieved:.1f}% anomaly, "
        f"rebalanced from the natural 2.93%) -> {dest}"
    )
    if len(anomaly_lines) < want_anomaly:
        print(f"  note: only {len(anomaly_lines)} anomalies exist in the train split")
    return counts


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the HDFS anomaly-classification training pipeline")
    parser.add_argument("--config", default=DEFAULT_TEMPLATE, help="Axolotl config template to render")
    parser.add_argument(
        "--max-train-samples", type=int, default=None,
        help="Train on this many examples instead of the full split (smoke test / staged run)",
    )
    parser.add_argument(
        "--train-anomaly-ratio", type=float, default=None,
        help=(
            "Target anomaly fraction of the --max-train-samples subset (e.g. 0.25). "
            "The real prior is 2.93%%, so an unweighted subset teaches the model that "
            "always answering 'normal' is 97%% correct and anomaly recall collapses. "
            "Only affects the training subset -- evaluation still runs on the untouched "
            "test split. Omit to keep the natural distribution."
        ),
    )
    args = parser.parse_args()

    for key, default in REQUIRED_ENV_DEFAULTS.items():
        os.environ.setdefault(key, default)

    # An unattended run must never block on a credential prompt: wandb without an
    # API key can sit waiting for input in a non-interactive container and stall
    # the entire job. Fail safe to tracking-disabled instead.
    if os.environ["USE_WANDB"].lower() == "true" and not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY is not set; disabling W&B for this run to avoid an interactive prompt")
        os.environ["USE_WANDB"] = "false"
        os.environ["WANDB_MODE"] = "disabled"

    max_train_samples = args.max_train_samples or os.environ.get("MAX_TRAIN_SAMPLES")
    template = args.config

    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "artifacts" / "hdfs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DIR.exists():
        raise SystemExit(
            f"{PROCESSED_DIR} not found. Run `python -m src.cli dataset prepare --name hdfs` first."
        )
    for required_file in ("train.jsonl", "validation_quick.jsonl", "test.jsonl"):
        if not (PROCESSED_DIR / required_file).exists():
            raise SystemExit(
                f"{PROCESSED_DIR / required_file} not found. Re-run "
                "`python -m src.cli dataset prepare --name hdfs` (a dataset built before "
                "validation_quick.jsonl was introduced will be missing it)."
            )

    # 1) Preflight.
    run([
        "python", "-m", "src.training.hdfs_preflight",
        "--base-model", os.environ["BASE_MODEL"],
        "--sequence-len", os.environ["SEQUENCE_LEN"],
        "--micro-batch-size", os.environ["MICRO_BATCH_SIZE"],
        "--grad-accumulation", os.environ["GRAD_ACCUMULATION"],
        "--output", str(run_dir / "preflight.json"),
    ])

    train_file = PROCESSED_DIR / "train.jsonl"
    if max_train_samples:
        train_file = run_dir / "train.subset.jsonl"
        write_train_subset(
            PROCESSED_DIR / "train.jsonl",
            train_file,
            int(max_train_samples),
            anomaly_ratio=args.train_anomaly_ratio,
            seed=int(os.environ["SEED"]),
        )
    max_train_samples = str(max_train_samples) if max_train_samples else None

    # 2) Render config.
    render_env = {
        **os.environ,
        "TRAIN_FILE": str(train_file),
        "VALID_FILE": str(PROCESSED_DIR / "validation.jsonl"),
        "QUICK_VALID_FILE": str(PROCESSED_DIR / "validation_quick.jsonl"),
        "OUTPUT_DIR": str(run_dir / "checkpoints"),
        "PREPARED_DIR": str(run_dir / "prepared"),
        "WANDB_RUN_NAME": os.environ.get("WANDB_RUN_NAME") or f"hdfs-{run_id}",
    }
    rendered_config = run_dir / "train.yml"
    run([
        "python", "-m", "src.training.render_config",
        "--template", template,
        "--output", str(rendered_config),
    ], env=render_env)

    # 3) Train.
    run(["axolotl", "train", str(rendered_config)])

    # 4) Locate adapter.
    adapter_dir = run_dir / "checkpoints"
    if not (adapter_dir / "adapter_config.json").exists():
        checkpoints = sorted(adapter_dir.glob("checkpoint-*"))
        if not checkpoints:
            raise SystemExit("No adapter found after training")
        adapter_dir = checkpoints[-1]

    # 5) Merge into a standalone model for serving/upload.
    # Deliberately NON-FATAL: the trained adapter is the valuable output and
    # evaluation/inference both load base+adapter directly, so a merge failure
    # must not throw away hours of completed training right before the eval
    # stage runs. It is reported loudly and recorded in run_metadata.json instead.
    #
    # Uses src.training.merge_adapter rather than `axolotl merge-lora --dequant`:
    # that path reloads the base in NF4 and dequantizes, which bakes ~9.5% mean
    # quantization error into weights whose LoRA delta is ~20x smaller than the
    # error itself. merge_adapter loads the base at full precision instead.
    merged_dir = run_dir / "checkpoints" / "merged"
    merge_error: str | None = None
    try:
        run([
            "python", "-m", "src.training.merge_adapter",
            "--adapter", str(adapter_dir),
            "--output", str(merged_dir),
            "--base-model", os.environ["BASE_MODEL"],
            "--force",
        ])
    except subprocess.CalledProcessError as exc:
        merge_error = f"merge_adapter failed: {exc}"
    if not merged_dir.exists() and merge_error is None:
        merge_error = f"merge_adapter reported success but {merged_dir} does not exist"

    if merge_error:
        print(f"WARNING: {merge_error}")
        print(f"         Training itself succeeded; the adapter is at {adapter_dir}")
        print("         Evaluation below falls back to base+adapter, but nothing can be uploaded.")
    else:
        final_dir = run_dir / "final"
        if not final_dir.exists():
            final_dir.symlink_to(merged_dir, target_is_directory=True)

    # 6) Evaluate on the held-out test split. Scores the merged model when there is
    # one, base+adapter otherwise, so a failed merge still produces numbers rather
    # than skipping the gate. Stratified by default: the real prior is 2.93% anomaly,
    # so a representative sample contains too few anomalies to measure recall at all.
    eval_dir = run_dir / "evaluation"
    eval_cmd = [
        "python", "-m", "src.evaluation.hdfs_evaluate",
        "--dataset", str(PROCESSED_DIR / "test.jsonl"),
        "--output-dir", str(eval_dir),
        "--stratified-anomalies", os.getenv("EVAL_ANOMALIES", "100"),
        "--stratified-normals", os.getenv("EVAL_NORMALS", "100"),
        "--max-new-tokens", os.getenv("EVAL_MAX_NEW_TOKENS", "512"),
    ]
    if merge_error:
        eval_cmd += ["--base-model", os.environ["BASE_MODEL"], "--adapter", str(adapter_dir)]
    else:
        eval_cmd += ["--base-model", str(merged_dir)]

    eval_error: str | None = None
    try:
        run(eval_cmd)
    except subprocess.CalledProcessError as exc:
        eval_error = f"evaluation failed: {exc}"
        print(f"WARNING: {eval_error}")

    # 7) Release gate. Fails closed -- no metrics means no promotion, so a crashed
    # evaluation can never be mistaken for a passing one.
    metrics_path = eval_dir / "metrics.json"
    gate: dict[str, Any] | None = None
    if metrics_path.exists():
        gate = gate_report(json.loads(metrics_path.read_text(encoding="utf-8")), thresholds_from_env())
        (run_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(format_report(gate))
    else:
        print(f"WARNING: no metrics at {metrics_path}; the release gate fails closed")
    gate_passed = bool(gate and gate["passed"])

    # 8) Reproducibility metadata.
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": os.environ["SEED"],
        "dataset_source": "LogHub HDFS_v1 (https://zenodo.org/records/8196385/files/HDFS_v1.zip)",
        "dataset_manifest": str(PROCESSED_DIR / "dataset_manifest.json"),
        "base_model": os.environ["BASE_MODEL"],
        "model_revision": os.environ["MODEL_REVISION"],
        "training_config": str(rendered_config),
        "max_train_samples": max_train_samples,
        "train_anomaly_ratio": args.train_anomaly_ratio,
        "train_file": str(train_file),
        "package_versions": collect_package_versions(),
        "python_version": sys.version,
        "git_commit": git_commit_hash(),
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir) if not merge_error else None,
        "merge_error": merge_error,
        "evaluation_dir": str(eval_dir),
        "eval_error": eval_error,
        "gate_passed": gate_passed,
        "gate": gate,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 9) Promote to the model registry. Gated deliberately: a run that did not clear
    # the eval thresholds must never reach the repo serving is pulled from.
    if not gate_passed:
        print("Release gate did not pass; skipping Hub upload")
    elif merge_error:
        print("No merged model to upload; skipping Hub upload")
    elif not os.environ.get("HF_MODEL_REPO"):
        print("HF_MODEL_REPO not set; skipping Hub upload")
    else:
        run(["python", "-m", "src.registry.upload_to_hub"], env={
            **os.environ,
            "MODEL_DIR": str(merged_dir),
            "EVAL_FILE": str(metrics_path),
            "MANIFEST_FILE": str(run_dir / "run_metadata.json"),
            "RUN_ID": run_id,
        })

    print(f"HDFS PIPELINE SUCCEEDED: {run_id}")
    print(f"Adapter: {adapter_dir}")
    if not merge_error:
        print(f"Merged model: {merged_dir}")
    print(f"Release gate: {'PASSED' if gate_passed else 'NOT PASSED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
