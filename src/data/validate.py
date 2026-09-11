from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def validate_record(record: dict[str, Any], line_no: int) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return [f"line {line_no}: 'messages' must be a non-empty list"]

    assistant_count = 0
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"line {line_no}, message {idx}: must be an object")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            errors.append(f"line {line_no}, message {idx}: invalid role={role!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"line {line_no}, message {idx}: content must be non-empty text")
        if role == "assistant":
            assistant_count += 1

    if assistant_count == 0:
        errors.append(f"line {line_no}: no assistant message found")
    return errors


def validate_file(path: Path, max_chars: int = 100_000) -> tuple[int, str]:
    seen: set[str] = set()
    count = 0
    errors: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_no}: root must be an object")
                continue

            fingerprint = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if fingerprint in seen:
                errors.append(f"line {line_no}: duplicate example")
            seen.add(fingerprint)

            errors.extend(validate_record(record, line_no))
            if len(raw) > max_chars:
                errors.append(f"line {line_no}: example exceeds {max_chars} characters")
            count += 1

    return count, "\n".join(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    count, error_text = validate_file(args.path)
    print(f"Validated {count} non-empty records from {args.path}")
    if error_text:
        print("DATASET VALIDATION FAILED")
        print(error_text)
        return 1
    print("DATASET VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
