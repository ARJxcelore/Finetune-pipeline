from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        normalized = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 records to create train/validation splits")
    if not 0 < args.valid_ratio < 0.5:
        raise SystemExit("--valid-ratio must be > 0 and < 0.5")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    valid_count = max(1, int(round(len(rows) * args.valid_ratio)))
    valid = rows[:valid_count]
    train = rows[valid_count:]

    if not train or not valid:
        raise SystemExit("Split produced an empty train or validation set")

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    manifest = {
        "input": str(args.input),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "total_records": len(rows),
        "train_records": len(train),
        "valid_records": len(valid),
        "dataset_fingerprint_sha256": fingerprint(rows),
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
