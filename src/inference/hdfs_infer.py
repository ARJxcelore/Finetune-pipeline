"""Single-trace HDFS incident classification inference.

Uses the same chat template, system prompt, and preprocessing as training and
evaluation (src.data.hdfs.sft_format, src.evaluation.hdfs_model) so inference
results are representative of the evaluation numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.hdfs.sft_format import SYSTEM_PROMPT
from src.evaluation.hdfs_model import generate, load_model
from src.evaluation.hdfs_schema import parse_and_validate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HDFS incident classification on a raw log trace")
    parser.add_argument("--model-path", required=True, help="Merged model dir, or base model id (see --base-model)")
    parser.add_argument("--base-model", default=None, help="Set if --model-path is a LoRA adapter, not a merged model")
    parser.add_argument("--input-file", type=Path, required=True, help="Text file with raw HDFS log lines for one trace")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    trace_text = args.input_file.read_text(encoding="utf-8").strip()
    if args.base_model:
        tokenizer, model = load_model(args.base_model, args.model_path)
    else:
        tokenizer, model = load_model(args.model_path, None)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": trace_text},
    ]
    raw_output = generate(tokenizer, model, messages, max_new_tokens=args.max_new_tokens)
    parsed = parse_and_validate(raw_output)

    if not parsed.schema_valid:
        print(json.dumps({
            "error": "model output failed schema validation",
            "details": parsed.errors,
            "parsed_json": parsed.json_obj,
            "raw_output": raw_output,
        }, indent=2))
        return 1

    print(json.dumps(parsed.json_obj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
