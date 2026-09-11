import json
from pathlib import Path

import pytest

from src.data.hdfs.schema import (
    extract_block_ids,
    load_anomaly_labels,
    reconstruct_traces,
    validate_join,
)
from src.data.hdfs.sft_format import build_sft_record, select_evidence
from src.data.hdfs.split import assert_no_overlap, split_records
from src.evaluation.hdfs_schema import evidence_grounding_rate, parse_and_validate
from src.evaluation.hdfs_metrics import classification_metrics, json_validity_rate


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---- dataset ----

def test_hdfs_schema_validation(tmp_path):
    label_csv = write(
        tmp_path / "anomaly_label.csv",
        "BlockId,Label\nblk_1,Normal\nblk_2,Anomaly\n",
    )
    labels = load_anomaly_labels(label_csv)
    assert labels == {"blk_1": "normal", "blk_2": "anomaly"}


def test_hdfs_schema_validation_rejects_bad_label(tmp_path):
    label_csv = write(tmp_path / "anomaly_label.csv", "BlockId,Label\nblk_1,Weird\n")
    with pytest.raises(ValueError):
        load_anomaly_labels(label_csv)


def test_extract_block_ids():
    assert extract_block_ids("081109 INFO dfs.DataNode: blk_-1608999687919862906 done") == [
        "blk_-1608999687919862906"
    ]
    assert extract_block_ids("no block here") == []


def test_trace_reconstruction(tmp_path):
    log = write(
        tmp_path / "HDFS.log",
        "081109 INFO Receiving block blk_1 src\n"
        "081109 INFO Receiving block blk_2 src\n"
        "081109 WARN blk_1 something\n",
    )
    labels = {"blk_1": "normal", "blk_2": "anomaly"}
    traces, stats = reconstruct_traces(log, labels)
    assert traces["blk_1"].raw_logs == [
        "081109 INFO Receiving block blk_1 src",
        "081109 WARN blk_1 something",
    ]
    assert traces["blk_1"].label == "normal"
    assert traces["blk_2"].label == "anomaly"
    assert stats["lines_seen"] == 3


def test_trace_label_join(tmp_path):
    log = write(tmp_path / "HDFS.log", "081109 INFO blk_1 ok\n081109 INFO blk_999 ok\n")
    labels = {"blk_1": "normal"}
    traces, stats = reconstruct_traces(log, labels)
    assert "blk_999" not in traces
    assert stats["lines_with_unlabeled_block"] == 1
    assert validate_join(traces, labels) == []


def _make_records(n_normal: int, n_anomaly: int) -> list[dict]:
    records = []
    for i in range(n_normal):
        records.append({"trace_id": f"norm_{i}", "label": "normal", "messages": []})
    for i in range(n_anomaly):
        records.append({"trace_id": f"anom_{i}", "label": "anomaly", "messages": []})
    return records


def test_no_split_overlap():
    records = _make_records(80, 20)
    train, valid, test = split_records(records, seed=42)
    assert len(train) + len(valid) + len(test) == 100
    assert_no_overlap(train, valid, test)  # should not raise


def test_split_is_deterministic():
    records = _make_records(80, 20)
    train1, valid1, test1 = split_records(records, seed=42)
    train2, valid2, test2 = split_records(records, seed=42)
    assert [r["trace_id"] for r in train1] == [r["trace_id"] for r in train2]
    assert [r["trace_id"] for r in valid1] == [r["trace_id"] for r in valid2]
    assert [r["trace_id"] for r in test1] == [r["trace_id"] for r in test2]


def test_no_label_leakage():
    trace_normal = type("T", (), {"trace_id": "blk_1", "raw_logs": ["INFO ok"], "label": "normal"})()
    trace_anomaly = type("T", (), {"trace_id": "blk_2", "raw_logs": ["ERROR bad"], "label": "anomaly"})()
    for trace in (trace_normal, trace_anomaly):
        record = build_sft_record(trace)
        user_content = next(m["content"] for m in record["messages"] if m["role"] == "user")
        assert "normal" not in user_content.lower()
        assert "anomaly" not in user_content.lower()
        assert "label" not in user_content.lower()


# ---- formatting ----

def test_sft_record_schema():
    trace = type("T", (), {"trace_id": "blk_1", "raw_logs": ["ERROR blk_1 failed", "WARN blk_1 retry"], "label": "anomaly"})()
    record = build_sft_record(trace)
    assert record["trace_id"] == "blk_1"
    roles = [m["role"] for m in record["messages"]]
    assert roles == ["system", "user", "assistant"]
    payload = json.loads(record["messages"][-1]["content"])
    assert payload["is_anomaly"] is True
    assert payload["incident_type"] == "anomaly"
    assert payload["evidence"]


def test_select_evidence_grounded_in_source():
    logs = [f"line {i}" for i in range(10)] + ["ERROR boom"]
    evidence = select_evidence(logs, max_lines=3)
    assert all(line in logs for line in evidence)
    assert "ERROR boom" in evidence


def test_output_json_schema():
    good = json.dumps({
        "is_anomaly": True, "incident_type": "anomaly", "confidence": 0.9,
        "evidence": ["ERROR x"], "summary": "bad",
    })
    parsed = parse_and_validate(good)
    assert parsed.errors == []
    assert parsed.schema_valid
    assert parsed.json_valid
    assert parsed.json_obj["incident_type"] == "anomaly"

    bad = json.dumps({"is_anomaly": "yes", "incident_type": "unknown", "confidence": 2, "evidence": "x", "summary": 1})
    parsed = parse_and_validate(bad)
    assert not parsed.schema_valid
    assert len(parsed.errors) >= 4


def test_json_valid_but_schema_invalid_are_distinguished():
    """The real base-model failure mode: well-formed JSON, out-of-range confidence,
    evidence as objects instead of strings. JSON validity and schema validity must
    not collapse into one flag, and the classification decision is still readable."""
    response = json.dumps({
        "is_anomaly": False,
        "incident_type": "normal",
        "confidence": 95,
        "evidence": [{"type": "data_received", "details": {}}],
        "summary": "ok",
    })
    parsed = parse_and_validate(response)
    assert parsed.json_valid is True
    assert parsed.schema_valid is False
    assert parsed.predicted_is_anomaly is False
    assert any("confidence" in e for e in parsed.errors)
    assert any("evidence" in e for e in parsed.errors)


def test_unparseable_output_has_no_prediction():
    parsed = parse_and_validate("I cannot determine whether this trace is anomalous.")
    assert parsed.json_valid is False
    assert parsed.schema_valid is False
    assert parsed.predicted_is_anomaly is None


# ---- evaluation ----

def test_prediction_parser_handles_prose_wrapping():
    text = 'Sure, here is the result:\n```json\n{"is_anomaly": false, "incident_type": "normal", "confidence": 1.0, "evidence": [], "summary": "ok"}\n```'
    parsed = parse_and_validate(text)
    assert parsed.errors == []
    assert parsed.json_obj["is_anomaly"] is False


def test_evidence_grounding_rate():
    trace = "line one\nERROR line two\nline three"
    assert evidence_grounding_rate(["ERROR line two"], trace) == 1.0
    assert evidence_grounding_rate(["ERROR line two", "invented line"], trace) == 0.5
    assert evidence_grounding_rate([], trace) == 1.0


def test_stratified_sampling_balances_classes(tmp_path):
    """A representative sample of this dataset yields too few anomalies to measure
    recall, so the evaluator must be able to over-sample the positive class."""
    from src.evaluation.hdfs_evaluate import load_stratified

    path = tmp_path / "test.jsonl"
    with path.open("w") as handle:
        for i in range(200):
            label = "anomaly" if i % 50 == 0 else "normal"  # 4 anomalies, 196 normal
            handle.write(json.dumps({"trace_id": f"blk_{i}", "label": label, "messages": []}) + "\n")

    rows = load_stratified(path, n_anomaly=3, n_normal=5, seed=42)
    assert sum(1 for r in rows if r["label"] == "anomaly") == 3
    assert sum(1 for r in rows if r["label"] == "normal") == 5

    # Asking for more anomalies than exist takes all of them rather than failing.
    rows = load_stratified(path, n_anomaly=99, n_normal=1, seed=42)
    assert sum(1 for r in rows if r["label"] == "anomaly") == 4

    # Deterministic for a fixed seed.
    a = [r["trace_id"] for r in load_stratified(path, 2, 2, seed=7)]
    b = [r["trace_id"] for r in load_stratified(path, 2, 2, seed=7)]
    assert a == b


def test_metric_calculation():
    records = [
        {"ground_truth_is_anomaly": True, "predicted_is_anomaly": True, "json_valid": True, "schema_valid": True},
        {"ground_truth_is_anomaly": True, "predicted_is_anomaly": False, "json_valid": True, "schema_valid": True},
        {"ground_truth_is_anomaly": False, "predicted_is_anomaly": False, "json_valid": True, "schema_valid": True},
        {"ground_truth_is_anomaly": False, "predicted_is_anomaly": None, "json_valid": False, "schema_valid": False},
    ]
    metrics = classification_metrics(records)
    assert metrics["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert metrics["accuracy"] == 0.5
    assert json_validity_rate(records) == 0.75
