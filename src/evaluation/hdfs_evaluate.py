"""Run the HDFS test set through a model (base or base+adapter) and score it.

Same prompt, same preprocessing, same generation settings regardless of
whether an adapter is supplied -- so base-model and fine-tuned runs are
directly comparable (see src.evaluation.hdfs_compare).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.evaluation.hdfs_metrics import summarize
from src.evaluation.hdfs_model import generate, last_assistant, load_model, prompt_messages, source_trace
from src.evaluation.hdfs_schema import evidence_grounding_rate, parse_and_validate


def load_dataset(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            rows.append(json.loads(raw))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_stratified(path: Path, n_anomaly: int, n_normal: int, seed: int) -> list[dict[str, Any]]:
    """Sample a fixed number of each class.

    Needed because the real class balance is 2.93% anomaly: a plain 300-row
    random sample contains ~9 anomalies, which makes anomaly precision/recall/F1
    statistically meaningless. Recall is unaffected by the sampling ratio, but
    PRECISION ON A STRATIFIED SAMPLE OVERSTATES real-world precision (the
    positive class is over-represented) -- the sampling metadata is recorded in
    metrics.json so this cannot be misread. Use --max-samples instead for a
    distribution-representative estimate.
    """
    anomalies: list[dict[str, Any]] = []
    normals: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            (anomalies if row.get("label") == "anomaly" else normals).append(row)

    rng = random.Random(seed)
    picked_anomaly = rng.sample(anomalies, min(n_anomaly, len(anomalies)))
    picked_normal = rng.sample(normals, min(n_normal, len(normals)))
    combined = picked_anomaly + picked_normal
    rng.shuffle(combined)
    print(
        f"Stratified sample: {len(picked_anomaly)} anomaly + {len(picked_normal)} normal "
        f"(available: {len(anomalies)} anomaly / {len(normals)} normal)"
    )
    return combined


def evaluate_row(tokenizer: Any, model: Any, row: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    messages = row["messages"]
    reference = last_assistant(messages)
    reference_parsed = parse_and_validate(reference)
    trace_text = source_trace(messages)

    prediction_text = generate(tokenizer, model, prompt_messages(messages), max_new_tokens=max_new_tokens)
    parsed = parse_and_validate(prediction_text)

    grounding = None
    if parsed.json_obj is not None:
        evidence = parsed.json_obj.get("evidence")
        if isinstance(evidence, list) and all(isinstance(e, str) for e in evidence):
            grounding = evidence_grounding_rate(evidence, trace_text)

    ground_truth = (
        bool(reference_parsed.json_obj["is_anomaly"])
        if reference_parsed.json_obj is not None
        else row.get("label") == "anomaly"
    )

    return {
        "trace_id": row.get("trace_id"),
        "input_trace": trace_text,
        "ground_truth_is_anomaly": ground_truth,
        "ground_truth_label": row.get("label"),
        "reference_output": reference,
        "prediction_raw": prediction_text,
        "predicted_is_anomaly": parsed.predicted_is_anomaly,
        "predicted_object": parsed.json_obj,
        "json_valid": parsed.json_valid,
        "schema_valid": parsed.schema_valid,
        "parse_errors": parsed.errors,
        "evidence_grounding_rate": grounding,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a model on the HDFS anomaly test set")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter; omit to evaluate the base model")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None, help="Distribution-representative sample (first N rows of the shuffled split)")
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help=(
            "Must exceed the longest expected target: anomaly targets need median 275 / "
            "p95 316 tokens because evidence quotes real log lines verbatim. Too small a "
            "budget truncates valid answers mid-JSON and they score as parse failures."
        ),
    )
    parser.add_argument(
        "--stratified-anomalies", type=int, default=None,
        help=(
            "Sample this many anomaly rows (plus --stratified-normals normal rows) instead "
            "of a representative sample. Needed to measure anomaly recall at all, since the "
            "real class balance is 2.93%% anomaly. Precision on a stratified sample is "
            "optimistic -- see the sampling block written into metrics.json."
        ),
    )
    parser.add_argument("--stratified-normals", type=int, default=None, help="Companion to --stratified-anomalies")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stratified_anomalies or args.stratified_normals:
        n_anomaly = args.stratified_anomalies or 0
        n_normal = args.stratified_normals if args.stratified_normals is not None else n_anomaly
        rows = load_stratified(args.dataset, n_anomaly, n_normal, args.seed)
        sampling = {
            "mode": "stratified",
            "requested_anomalies": n_anomaly,
            "requested_normals": n_normal,
            "true_class_prior_anomaly_pct": 2.93,
            "warning": (
                "Anomaly recall is unaffected by stratification, but anomaly PRECISION and "
                "overall ACCURACY are optimistic here because anomalies are over-represented "
                "relative to the true 2.93% prior. Re-run with --max-samples for a "
                "distribution-representative estimate."
            ),
        }
    else:
        rows = load_dataset(args.dataset, args.max_samples)
        sampling = {"mode": "representative", "max_samples": args.max_samples}

    if not rows:
        raise SystemExit(f"No rows loaded from {args.dataset}")

    tokenizer, model = load_model(args.base_model, args.adapter)

    results = [evaluate_row(tokenizer, model, row, args.max_new_tokens) for row in rows]
    metrics = summarize(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    metrics_payload = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset": str(args.dataset),
        "samples": len(results),
        "sampling": sampling,
        **metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(json.dumps(metrics_payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
