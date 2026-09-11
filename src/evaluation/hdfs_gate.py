"""Release gate for HDFS anomaly-classification runs.

Turns a metrics.json from src.evaluation.hdfs_evaluate into a pass/fail
decision so an unattended run self-certifies, instead of depending on someone
reading the numbers the next morning. Nothing reaches the model registry unless
this passes.

Defaults sit ~10 points below the observed fine-tuned baseline (schema 100%,
accuracy 0.995, anomaly recall 0.99, anomaly F1 0.995, grounding 0.999): loose
enough to absorb run-to-run variance, tight enough to catch the two failure
modes this repo has actually produced -- an unadapted base model (schema 0%,
anomaly recall 0.0) and a truncated-output run (schema 71.5%, caused by too
small an EVAL_MAX_NEW_TOKENS budget).

Anomaly recall and anomaly F1 are both checked on purpose. Recall alone is the
operationally important number -- a missed incident costs more than a false
alarm -- but a model that simply answers "anomaly" every time scores recall 1.0,
and only F1 catches it.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping

# metric key -> (env var, default threshold)
THRESHOLDS: dict[str, tuple[str, float]] = {
    "valid_schema_pct": ("HDFS_GATE_MIN_SCHEMA_VALID_PCT", 90.0),
    "accuracy": ("HDFS_GATE_MIN_ACCURACY", 0.90),
    "anomaly_recall": ("HDFS_GATE_MIN_ANOMALY_RECALL", 0.85),
    "anomaly_f1": ("HDFS_GATE_MIN_ANOMALY_F1", 0.85),
    "evidence_grounding_rate": ("HDFS_GATE_MIN_EVIDENCE_GROUNDING", 0.90),
}


@dataclass(frozen=True)
class Check:
    metric: str
    value: float
    threshold: float
    passed: bool


def extract_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the nested summarize() payload onto the gate's metric names."""
    anomaly = metrics.get("anomaly") or {}
    return {
        "valid_schema_pct": float(metrics.get("valid_schema_pct", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "anomaly_recall": float(anomaly.get("recall", 0.0)),
        "anomaly_f1": float(anomaly.get("f1", 0.0)),
        "evidence_grounding_rate": float(metrics.get("evidence_grounding_rate", 0.0)),
    }


def thresholds_from_env(env: Mapping[str, str] | None = None) -> dict[str, float]:
    env = os.environ if env is None else env
    return {metric: float(env.get(var, default)) for metric, (var, default) in THRESHOLDS.items()}


def run_gate(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> tuple[bool, list[Check]]:
    values = extract_values(metrics)
    checks = [
        Check(metric=metric, value=values[metric], threshold=thresholds[metric], passed=values[metric] >= thresholds[metric])
        for metric in THRESHOLDS
    ]
    return all(check.passed for check in checks), checks


def gate_report(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    passed, checks = run_gate(metrics, thresholds)
    return {
        "passed": passed,
        "checks": [asdict(check) for check in checks],
        "failures": [check.metric for check in checks if not check.passed],
        "samples": metrics.get("samples"),
        # Stratified sampling inflates precision and accuracy relative to the real
        # 2.93% anomaly prior; carried through so a passing gate cannot be misread.
        "sampling": metrics.get("sampling"),
    }


def format_report(report: Mapping[str, Any]) -> str:
    lines = [f"{'PASS' if report['passed'] else 'FAIL'}  HDFS release gate"]
    for check in report["checks"]:
        mark = "ok  " if check["passed"] else "FAIL"
        lines.append(f"  {mark} {check['metric']:<26} {check['value']:.4f}  (min {check['threshold']})")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the release gate to an HDFS metrics.json")
    parser.add_argument("--metrics", type=Path, required=True, help="metrics.json from src.evaluation.hdfs_evaluate")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the gate report JSON")
    args = parser.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    report = gate_report(metrics, thresholds_from_env())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
