"""Trace-level (block-level) 80/10/10 split with leakage checks.

Log lines from the same trace/block must never be split across
train/validation/test -- since our unit is already one whole trace per
record, this reduces to: shuffle trace ids with a fixed seed, then slice.
"""

from __future__ import annotations

import random
from typing import Any


def split_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be in (0, 1)")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1 (remainder is the test split)")

    ordered = sorted(records, key=lambda r: r["trace_id"])
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    train_end = int(round(total * train_ratio))
    valid_end = train_end + int(round(total * valid_ratio))

    train = ordered[:train_end]
    valid = ordered[train_end:valid_end]
    test = ordered[valid_end:]

    if not train or not valid or not test:
        raise ValueError("Split produced an empty split; dataset too small for 80/10/10")

    assert_no_overlap(train, valid, test)
    return train, valid, test


def assert_no_overlap(
    train: list[dict[str, Any]], valid: list[dict[str, Any]], test: list[dict[str, Any]]
) -> None:
    train_ids = {r["trace_id"] for r in train}
    valid_ids = {r["trace_id"] for r in valid}
    test_ids = {r["trace_id"] for r in test}

    overlaps = {
        "train_valid": train_ids & valid_ids,
        "train_test": train_ids & test_ids,
        "valid_test": valid_ids & test_ids,
    }
    problems = {name: ids for name, ids in overlaps.items() if ids}
    if problems:
        raise ValueError(f"Split overlap detected: {problems}")
