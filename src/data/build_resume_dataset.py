from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

TURN_PATTERN = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.S)
THINK_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
RESUME_PREFIX = re.compile(r"^Resume:\s*\n?")


def parse_row(text: str) -> dict[str, str] | None:
    matches = TURN_PATTERN.findall(text)
    if [role for role, _ in matches] != ["system", "user", "assistant"]:
        return None
    system_c, user_c, assistant_c = (content for _role, content in matches)
    user_c = RESUME_PREFIX.sub("", user_c).strip()
    assistant_c = THINK_PATTERN.sub("", assistant_c).strip()
    try:
        json.loads(assistant_c)
    except json.JSONDecodeError:
        return None
    return {"system": system_c.strip(), "user": user_c, "assistant": assistant_c}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="sandeeppanem/resume-json-extraction-5k parquet file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct", help="tokenizer used to enforce --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=2560, help="drops examples whose full chat-template token count exceeds this")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    df = pd.read_parquet(args.input)

    records: list[dict[str, list[dict[str, str]]]] = []
    seen: set[str] = set()
    for text in df["text"]:
        parsed = parse_row(text)
        if parsed is None:
            continue
        if parsed["user"] in seen:
            continue
        seen.add(parsed["user"])

        messages = [
            {"role": "system", "content": parsed["system"]},
            {"role": "user", "content": parsed["user"]},
            {"role": "assistant", "content": parsed["assistant"]},
        ]
        token_count = len(tokenizer.apply_chat_template(messages, tokenize=True))
        if token_count > args.max_tokens:
            continue
        records.append({"messages": messages})

    rng = random.Random(args.seed)
    rng.shuffle(records)
    sampled = records[: args.sample_size]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"kept {len(records)} usable records (<= {args.max_tokens} tokens), sampled {len(sampled)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
