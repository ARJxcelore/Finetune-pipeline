"""Compare base-model vs fine-tuned predictions on the same test set and build a qualitative report.

Expects two predictions.jsonl files produced by src.evaluation.hdfs_evaluate,
run over the exact same --dataset (same order of trace_ids), so that row i
in one file corresponds to row i in the other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.hdfs_metrics import summarize

QUALITATIVE_BUCKET_SIZE = 5


def load_predictions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket_name(ground_truth: bool, predicted: bool | None) -> str:
    if ground_truth and predicted is True:
        return "true_positive"
    if not ground_truth and predicted is False:
        return "true_negative"
    if not ground_truth and predicted is not False:
        return "false_positive"
    return "false_negative"


def build_qualitative_report(
    base_results: list[dict[str, Any]],
    finetuned_results: list[dict[str, Any]],
    *,
    bucket_size: int = QUALITATIVE_BUCKET_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    if len(base_results) != len(finetuned_results):
        raise ValueError("base and fine-tuned predictions must cover the same rows (same length)")

    buckets: dict[str, list[dict[str, Any]]] = {
        "true_positive": [], "true_negative": [], "false_positive": [], "false_negative": [],
    }

    for base_row, ft_row in zip(base_results, finetuned_results):
        if base_row.get("trace_id") != ft_row.get("trace_id"):
            raise ValueError(
                f"row mismatch: base trace_id={base_row.get('trace_id')} vs "
                f"finetuned trace_id={ft_row.get('trace_id')}; predictions files must be aligned"
            )
        truth = ft_row["ground_truth_is_anomaly"]
        name = bucket_name(truth, ft_row["predicted_is_anomaly"])
        if len(buckets[name]) >= bucket_size:
            continue
        buckets[name].append({
            "trace_id": ft_row.get("trace_id"),
            "ground_truth_label": ft_row.get("ground_truth_label"),
            "input_trace": ft_row.get("input_trace"),
            "base_model_prediction": base_row.get("predicted_object") or base_row.get("prediction_raw"),
            "finetuned_prediction": ft_row.get("predicted_object") or ft_row.get("prediction_raw"),
            "correct": name in ("true_positive", "true_negative"),
        })

    return buckets


def render_markdown(comparison: dict[str, Any], qualitative: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# HDFS anomaly classification: base vs fine-tuned", ""]
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | base | fine-tuned |")
    lines.append("|---|---|---|")
    base_m, ft_m = comparison["base"], comparison["finetuned"]
    rows = [
        ("valid JSON %", "valid_json_pct"),
        ("valid schema %", "valid_schema_pct"),
        ("accuracy", "accuracy"),
        ("anomaly precision", ("anomaly", "precision")),
        ("anomaly recall", ("anomaly", "recall")),
        ("anomaly F1", ("anomaly", "f1")),
        ("normal precision", ("normal", "precision")),
        ("normal recall", ("normal", "recall")),
        ("evidence grounding rate", "evidence_grounding_rate"),
    ]
    for label, key in rows:
        def get(metrics: dict[str, Any]) -> Any:
            if isinstance(key, tuple):
                return metrics[key[0]][key[1]]
            return metrics[key]
        lines.append(f"| {label} | {get(base_m):.4f} | {get(ft_m):.4f} |")

    for bucket_key, title in [
        ("true_positive", "True positives"),
        ("true_negative", "True negatives"),
        ("false_positive", "False positives"),
        ("false_negative", "False negatives"),
    ]:
        lines += ["", f"## {title}", ""]
        for example in qualitative[bucket_key]:
            lines.append(f"### {example['trace_id']} (ground truth: {example['ground_truth_label']})")
            lines.append(f"- base model: `{example['base_model_prediction']}`")
            lines.append(f"- fine-tuned: `{example['finetuned_prediction']}`")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned HDFS predictions")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--finetuned-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_results = load_predictions(args.base_predictions)
    finetuned_results = load_predictions(args.finetuned_predictions)

    comparison = {
        "base": summarize(base_results),
        "finetuned": summarize(finetuned_results),
    }
    qualitative = build_qualitative_report(base_results, finetuned_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "qualitative_report.json").write_text(
        json.dumps(qualitative, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "qualitative_report.md").write_text(
        render_markdown(comparison, qualitative), encoding="utf-8"
    )
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
