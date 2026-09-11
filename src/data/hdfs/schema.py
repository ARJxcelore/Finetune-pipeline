"""Parsing and trace reconstruction for the LogHub HDFS_v1 dataset.

HDFS_v1 ships raw console logs (`HDFS.log`) plus preprocessed helper files
(`anomaly_label.csv`, `Event_traces.csv`, `HDFS_templates.csv`). Every raw log
line carries a `blk_<id>` (or `blk_-<id>`) token that identifies the HDFS block
the line is about; `anomaly_label.csv` maps each block id to a `Normal`/
`Anomaly` label. Traces are reconstructed directly from the raw log lines
(grouped by block id, in file order) rather than from the event-id-only
helper files, so the model sees actual log text instead of `E1 E2 E3`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

BLOCK_ID_RE = re.compile(r"blk_-?\d+")

Label = Literal["normal", "anomaly"]


@dataclass
class HDFSTrace:
    trace_id: str
    raw_logs: list[str]
    label: Label
    event_ids: list[str] = field(default_factory=list)


def extract_block_ids(line: str) -> list[str]:
    """Return all distinct block ids referenced by a raw log line, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in BLOCK_ID_RE.finditer(line):
        seen.setdefault(match.group(0), None)
    return list(seen.keys())


def load_anomaly_labels(path: Path) -> dict[str, Label]:
    """Parse anomaly_label.csv into {block_id: 'normal'|'anomaly'}.

    The official file has a header with a block-id column (observed name
    'BlockId') and a label column (observed name 'Label', values 'Normal' /
    'Anomaly'). Column names are resolved case-insensitively instead of
    assumed, and the file is validated rather than trusted blindly.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc

        lower_header = [col.strip().lower() for col in header]
        try:
            block_col = next(i for i, col in enumerate(lower_header) if "block" in col)
        except StopIteration as exc:
            raise ValueError(f"{path}: no block-id column found in header {header}") from exc
        try:
            label_col = next(i for i, col in enumerate(lower_header) if "label" in col)
        except StopIteration as exc:
            raise ValueError(f"{path}: no label column found in header {header}") from exc

        labels: dict[str, Label] = {}
        valid_raw_labels = {"normal": "normal", "anomaly": "anomaly"}
        for row_no, row in enumerate(reader, start=2):
            if not row:
                continue
            block_id = row[block_col].strip()
            raw_label = row[label_col].strip().lower()
            if raw_label not in valid_raw_labels:
                raise ValueError(
                    f"{path}:{row_no}: unrecognized label {raw_label!r} for block {block_id!r}"
                )
            if block_id in labels and labels[block_id] != valid_raw_labels[raw_label]:
                raise ValueError(f"{path}:{row_no}: conflicting labels for block {block_id!r}")
            labels[block_id] = valid_raw_labels[raw_label]

    if not labels:
        raise ValueError(f"{path}: no labels parsed")
    return labels


def iter_raw_log_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                yield line


def reconstruct_traces(
    log_path: Path,
    labels: dict[str, Label],
    *,
    max_lines: int | None = None,
) -> tuple[dict[str, HDFSTrace], dict[str, int]]:
    """Group raw HDFS.log lines by block id, in file order, and attach the ground-truth label.

    Returns (traces_by_block_id, stats) where stats reports lines seen,
    lines with no block id, and lines whose block id has no label.
    """
    traces: dict[str, list[str]] = {}
    stats = {"lines_seen": 0, "lines_without_block_id": 0, "lines_with_unlabeled_block": 0}

    for line in iter_raw_log_lines(log_path):
        stats["lines_seen"] += 1
        if max_lines is not None and stats["lines_seen"] > max_lines:
            break
        block_ids = extract_block_ids(line)
        if not block_ids:
            stats["lines_without_block_id"] += 1
            continue
        for block_id in block_ids:
            if block_id not in labels:
                stats["lines_with_unlabeled_block"] += 1
                continue
            traces.setdefault(block_id, []).append(line)

    result = {
        block_id: HDFSTrace(trace_id=block_id, raw_logs=lines, label=labels[block_id])
        for block_id, lines in traces.items()
    }
    return result, stats


def validate_join(traces: dict[str, HDFSTrace], labels: dict[str, Label]) -> list[str]:
    """Sanity-check the raw-log <-> label join. Returns a list of human-readable problems."""
    problems: list[str] = []

    labeled_but_absent = set(labels) - set(traces)
    if labeled_but_absent:
        problems.append(
            f"{len(labeled_but_absent)} labeled block ids never appear in the raw log "
            f"(example: {sorted(labeled_but_absent)[:3]})"
        )

    empty_traces = [trace_id for trace_id, trace in traces.items() if not trace.raw_logs]
    if empty_traces:
        problems.append(f"{len(empty_traces)} traces have zero raw log lines")

    bad_labels = [t.trace_id for t in traces.values() if t.label not in ("normal", "anomaly")]
    if bad_labels:
        problems.append(f"{len(bad_labels)} traces have a label outside {{normal, anomaly}}")

    return problems
