"""Dataset statistics for the HDFS SFT dataset (artifacts/hdfs/processed/dataset_stats.json)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def _length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": float(ordered[-1]),
    }


def compute_dataset_stats(
    records: list[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    total = len(records)
    normal = sum(1 for r in records if r["label"] == "normal")
    anomaly = total - normal

    trace_line_counts: list[int] = []
    char_counts: list[int] = []
    token_counts: list[int] = []

    for record in records:
        user_content = next(m["content"] for m in record["messages"] if m["role"] == "user")
        trace_line_counts.append(len(user_content.splitlines()))
        char_counts.append(len(user_content))
        if tokenizer is not None:
            token_counts.append(len(tokenizer(user_content, add_special_tokens=False)["input_ids"]))

    payload: dict[str, Any] = {
        "total_traces": total,
        "normal_traces": normal,
        "anomalous_traces": anomaly,
        "normal_pct": (normal / total * 100) if total else 0.0,
        "anomaly_pct": (anomaly / total * 100) if total else 0.0,
        "trace_length_lines": _length_stats(trace_line_counts),
        "trace_length_chars": _length_stats(char_counts),
    }

    if tokenizer is not None:
        payload["trace_length_tokens"] = _length_stats(token_counts)
        if max_seq_length is not None:
            over_limit = sum(1 for c in token_counts if c > max_seq_length)
            payload["tokens_over_max_seq_length"] = {
                "max_seq_length": max_seq_length,
                "count": over_limit,
                "pct": (over_limit / total * 100) if total else 0.0,
            }

    return payload


def human_readable_summary(stats: dict[str, Any]) -> str:
    lines = [
        "HDFS SFT dataset summary",
        "========================",
        f"total traces:      {stats['total_traces']}",
        f"normal traces:     {stats['normal_traces']} ({stats['normal_pct']:.2f}%)",
        f"anomalous traces:  {stats['anomalous_traces']} ({stats['anomaly_pct']:.2f}%)",
        "",
        "trace length (log lines):",
        f"  mean={stats['trace_length_lines']['mean']:.1f} median={stats['trace_length_lines']['median']:.1f} "
        f"p90={stats['trace_length_lines']['p90']:.1f} p95={stats['trace_length_lines']['p95']:.1f} "
        f"p99={stats['trace_length_lines']['p99']:.1f} max={stats['trace_length_lines']['max']:.0f}",
        "",
        "trace length (characters):",
        f"  mean={stats['trace_length_chars']['mean']:.1f} median={stats['trace_length_chars']['median']:.1f} "
        f"max={stats['trace_length_chars']['max']:.0f}",
    ]
    if "trace_length_tokens" in stats:
        tok = stats["trace_length_tokens"]
        lines += [
            "",
            "trace length (Qwen tokens, user content only):",
            f"  p50={tok['median']:.1f} p90={tok['p90']:.1f} p95={tok['p95']:.1f} "
            f"p99={tok['p99']:.1f} max={tok['max']:.0f}",
        ]
    if "tokens_over_max_seq_length" in stats:
        over = stats["tokens_over_max_seq_length"]
        lines += [
            "",
            f"traces exceeding max_seq_length={over['max_seq_length']}: "
            f"{over['count']} ({over['pct']:.2f}%)",
        ]
    return "\n".join(lines)


def write_stats(records: list[dict[str, Any]], output_path: Path, **kwargs: Any) -> dict[str, Any]:
    stats = compute_dataset_stats(records, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
