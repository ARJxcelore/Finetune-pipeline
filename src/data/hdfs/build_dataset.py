"""Build the HDFS SFT dataset: raw log + labels -> validated train/valid/test JSONL.

    download (src.data.hdfs.download)
        v
    parse anomaly_label.csv + HDFS.log, reconstruct traces (src.data.hdfs.schema)
        v
    validate the raw-log <-> label join
        v
    convert to SFT records (src.data.hdfs.sft_format)
        v
    trace-level 80/10/10 split with overlap checks (src.data.hdfs.split)
        v
    dataset statistics (src.data.hdfs.stats)
        v
    write artifacts/hdfs/processed/{train,validation,test}.jsonl + dataset_stats.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.data.hdfs.schema import load_anomaly_labels, reconstruct_traces, validate_join
from src.data.hdfs.sft_format import build_sft_record
from src.data.hdfs.split import split_records
from src.data.hdfs.stats import human_readable_summary, write_stats


def find_extracted_file(extract_dir: Path, name: str) -> Path:
    matches = list(extract_dir.rglob(name))
    if not matches:
        raise SystemExit(f"Could not find {name} under {extract_dir}. Run dataset download first.")
    return matches[0]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HDFS anomaly-classification SFT dataset")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/hdfs"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hdfs/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-log-lines", type=int, default=None,
        help="Only read this many lines of HDFS.log (smoke testing / fast iteration)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Used only to report how many traces exceed this many tokens; does not truncate",
    )
    parser.add_argument(
        "--no-tokenizer-stats", action="store_true",
        help="Skip Qwen-tokenizer-based token length stats (faster, no model download)",
    )
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--quick-valid-size", type=int, default=1000,
        help=(
            "Size of a fixed random subsample of the validation split, written as "
            "validation_quick.jsonl for Axolotl's periodic training-time eval "
            "(checkpoint selection only) -- evaluating the full validation split "
            "every eval_steps is impractically slow at real dataset scale. The "
            "rigorous evaluation is src.evaluation.hdfs_evaluate against test.jsonl."
        ),
    )
    args = parser.parse_args()

    extract_dir = args.raw_dir / "extracted"
    label_path = find_extracted_file(extract_dir, "anomaly_label.csv")
    log_path = find_extracted_file(extract_dir, "HDFS.log")

    print(f"Parsing labels from {label_path}")
    labels = load_anomaly_labels(label_path)
    print(f"  {len(labels)} labeled blocks")

    print(f"Reconstructing traces from {log_path} (raw log text, not just event ids)")
    traces, recon_stats = reconstruct_traces(log_path, labels, max_lines=args.max_log_lines)
    print(f"  reconstruction stats: {recon_stats}")
    print(f"  {len(traces)} traces reconstructed")

    problems = validate_join(traces, labels)
    if problems and args.max_log_lines is None:
        raise SystemExit("Raw-log <-> label join failed validation:\n" + "\n".join(problems))
    elif problems:
        print("Join validation warnings (expected with --max-log-lines truncation):")
        for problem in problems:
            print(f"  - {problem}")

    records = [build_sft_record(trace) for trace in traces.values()]
    for record in records:
        for message in record["messages"]:
            assert "label" not in message["content"].lower() or message["role"] == "assistant", (
                "possible label leakage into a non-assistant message"
            )

    train, valid, test = split_records(
        records, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, seed=args.seed
    )
    print(f"Split: train={len(train)} validation={len(valid)} test={len(test)}")

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)

    quick_valid_rng = random.Random(args.seed)
    quick_valid_size = min(args.quick_valid_size, len(valid))
    quick_valid = quick_valid_rng.sample(valid, quick_valid_size)
    write_jsonl(args.output_dir / "validation_quick.jsonl", quick_valid)
    print(f"Wrote {quick_valid_size}-row validation_quick.jsonl for training-time eval")

    tokenizer = None
    if not args.no_tokenizer_stats:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    stats = write_stats(
        records,
        args.output_dir / "dataset_stats.json",
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    print(human_readable_summary(stats))

    manifest = {
        "raw_dir": str(args.raw_dir),
        "label_file": str(label_path),
        "log_file": str(log_path),
        "reconstruction_stats": recon_stats,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "total_records": len(records),
        "train_records": len(train),
        "validation_records": len(valid),
        "test_records": len(test),
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote dataset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
