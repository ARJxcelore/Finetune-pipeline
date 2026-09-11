"""Classification metrics for HDFS anomaly-classification predictions."""

from __future__ import annotations

from typing import Any


def _safe_div(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def confusion_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    """records: each has ground_truth_is_anomaly: bool, predicted_is_anomaly: bool | None.

    A None prediction (unparseable output) counts as a miss against whichever
    class it should have predicted, i.e. it is never counted as a "correct" hit.
    """
    tp = fp = tn = fn = 0
    for record in records:
        truth = record["ground_truth_is_anomaly"]
        pred = record["predicted_is_anomaly"]
        if truth and pred is True:
            tp += 1
        elif truth and pred is not True:
            fn += 1
        elif not truth and pred is False:
            tn += 1
        else:
            fp += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = confusion_counts(records)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    total = tp + fp + tn + fn

    anomaly_precision = _safe_div(tp, tp + fp)
    anomaly_recall = _safe_div(tp, tp + fn)
    anomaly_f1 = _safe_div(2 * anomaly_precision * anomaly_recall, anomaly_precision + anomaly_recall)

    normal_precision = _safe_div(tn, tn + fn)
    normal_recall = _safe_div(tn, tn + fp)
    normal_f1 = _safe_div(2 * normal_precision * normal_recall, normal_precision + normal_recall)

    accuracy = _safe_div(tp + tn, total)

    return {
        "n": total,
        "confusion_matrix": counts,
        "accuracy": accuracy,
        "anomaly": {"precision": anomaly_precision, "recall": anomaly_recall, "f1": anomaly_f1},
        "normal": {"precision": normal_precision, "recall": normal_recall, "f1": normal_f1},
    }


def json_validity_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["json_valid"]) / len(records)


def schema_validity_rate(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r["schema_valid"]) / len(records)


def mean_evidence_grounding_rate(records: list[dict[str, Any]]) -> float:
    grounded = [r["evidence_grounding_rate"] for r in records if r.get("evidence_grounding_rate") is not None]
    return sum(grounded) / len(grounded) if grounded else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "valid_json_pct": json_validity_rate(records) * 100,
        "valid_schema_pct": schema_validity_rate(records) * 100,
        "evidence_grounding_rate": mean_evidence_grounding_rate(records),
        **classification_metrics(records),
    }
