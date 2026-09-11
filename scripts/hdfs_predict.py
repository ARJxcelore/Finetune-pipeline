#!/usr/bin/env python
"""Run the fine-tuned HDFS anomaly classifier locally and print the JSON verdict.

Three ways to use it -- the interactive mode exists because loading a 4-bit model
takes ~40 s, so a one-shot-per-trace CLI wastes that on every single call:

    # interactive: load once, classify many traces
    python scripts/hdfs_predict.py

    # one trace from a file
    python scripts/hdfs_predict.py --input-file examples/hdfs_trace.txt

    # piped in
    cat trace.txt | python scripts/hdfs_predict.py

    # a whole JSONL test split, one verdict per line
    python scripts/hdfs_predict.py --batch-file artifacts/hdfs/processed/test.jsonl --limit 20

Model loading, the chat template, the system prompt and the JSON validation are all
imported from the same modules training and evaluation use, so a verdict here is
exactly what the reported metrics measured -- not a re-implementation that could drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hdfs.sft_format import SYSTEM_PROMPT  # noqa: E402
from src.evaluation.hdfs_model import generate, load_model  # noqa: E402
from src.evaluation.hdfs_schema import evidence_grounding_rate, parse_and_validate  # noqa: E402

DEFAULT_ADAPTER = "artifacts/hdfs/runs/hdfs-20260909-132117/checkpoints"
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def classify(tokenizer, model, trace: str, max_new_tokens: int) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": trace.strip()},
    ]
    started = time.time()
    raw = generate(tokenizer, model, messages, max_new_tokens=max_new_tokens)
    elapsed = time.time() - started
    parsed = parse_and_validate(raw)

    out: dict = {"latency_seconds": round(elapsed, 2)}
    if parsed.json_obj is None:
        out.update({"error": "model output was not parseable JSON", "raw_output": raw})
        return out

    out.update(parsed.json_obj)
    out["schema_valid"] = parsed.schema_valid
    if not parsed.schema_valid:
        out["schema_errors"] = parsed.errors
    evidence = parsed.json_obj.get("evidence")
    if isinstance(evidence, list) and all(isinstance(e, str) for e in evidence):
        # Independently verify the model quoted real lines instead of inventing them.
        out["evidence_grounding_rate"] = evidence_grounding_rate(evidence, trace)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter dir (default: the trained run)")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--input-file", type=Path, help="A file holding one trace's log lines")
    parser.add_argument("--batch-file", type=Path, help="A JSONL split (test.jsonl); classifies each row's trace")
    parser.add_argument("--limit", type=int, default=None, help="With --batch-file, stop after N rows")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-adapter", action="store_true", help="Run the untuned base model, for comparison")
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    if adapter and not (ROOT / adapter / "adapter_config.json").exists() and not Path(adapter).exists():
        raise SystemExit(f"adapter not found: {adapter}")

    label = "base model (no adapter)" if adapter is None else f"adapter {adapter}"
    print(f"loading {args.base_model} in 4-bit with {label} ...", file=sys.stderr)
    tokenizer, model = load_model(args.base_model, adapter)
    print("ready\n", file=sys.stderr)

    if args.batch_file:
        rows = [json.loads(line) for line in args.batch_file.open() if line.strip()]
        if args.limit:
            rows = rows[: args.limit]
        correct = 0
        for i, row in enumerate(rows, 1):
            trace = next(m["content"] for m in row["messages"] if m["role"] == "user")
            verdict = classify(tokenizer, model, trace, args.max_new_tokens)
            truth = row.get("label")
            got = "anomaly" if verdict.get("is_anomaly") else "normal"
            hit = truth == got
            correct += hit
            print(f"[{i}/{len(rows)}] {row.get('trace_id')}  truth={truth:7s} predicted={got:7s} "
                  f"{'OK' if hit else 'MISS'}  ({verdict['latency_seconds']}s)")
        print(f"\n{correct}/{len(rows)} correct")
        return 0

    if args.input_file:
        print(json.dumps(classify(tokenizer, model, args.input_file.read_text(), args.max_new_tokens), indent=2))
        return 0

    if not sys.stdin.isatty():
        print(json.dumps(classify(tokenizer, model, sys.stdin.read(), args.max_new_tokens), indent=2))
        return 0

    print("Paste HDFS log lines for ONE block, then a blank line to classify. Ctrl-D to quit.")
    while True:
        lines: list[str] = []
        try:
            while True:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
        except EOFError:
            print("\nbye")
            return 0
        if not lines:
            continue
        print(json.dumps(classify(tokenizer, model, "\n".join(lines), args.max_new_tokens), indent=2))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
