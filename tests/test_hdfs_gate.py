import json
from pathlib import Path

from src.evaluation.hdfs_gate import gate_report, run_gate, thresholds_from_env

DEFAULTS = thresholds_from_env({})


def metrics(*, schema=100.0, accuracy=0.995, recall=0.99, f1=0.995, grounding=0.999):
    return {
        "valid_schema_pct": schema,
        "accuracy": accuracy,
        "evidence_grounding_rate": grounding,
        "anomaly": {"precision": 1.0, "recall": recall, "f1": f1},
    }


def test_observed_finetuned_run_passes():
    passed, _ = run_gate(metrics(), DEFAULTS)
    assert passed


def test_unadapted_base_model_fails():
    passed, checks = run_gate(
        metrics(schema=0.0, accuracy=0.495, recall=0.0, f1=0.0, grounding=0.0), DEFAULTS
    )
    assert not passed
    assert {c.metric for c in checks if not c.passed} == set(DEFAULTS)


def test_truncated_output_run_fails_on_schema():
    """The maxtok256 regression: valid answers cut mid-JSON score as parse failures."""
    passed, checks = run_gate(metrics(schema=71.5), DEFAULTS)
    assert not passed
    assert [c.metric for c in checks if not c.passed] == ["valid_schema_pct"]


def test_always_anomaly_model_fails_despite_perfect_recall():
    """Recall alone is gameable; F1 and accuracy are what catch this."""
    passed, checks = run_gate(metrics(accuracy=0.5, recall=1.0, f1=0.667), DEFAULTS)
    failed = {c.metric for c in checks if not c.passed}
    assert not passed
    assert "anomaly_recall" not in failed
    assert failed == {"accuracy", "anomaly_f1"}


def test_missing_metrics_keys_fail_closed():
    passed, _ = run_gate({}, DEFAULTS)
    assert not passed


def test_env_overrides_threshold():
    relaxed = thresholds_from_env({"HDFS_GATE_MIN_SCHEMA_VALID_PCT": "70"})
    assert relaxed["valid_schema_pct"] == 70.0
    assert run_gate(metrics(schema=71.5), relaxed)[0]


def test_report_carries_sampling_block():
    sampling = {"mode": "stratified", "true_class_prior_anomaly_pct": 2.93}
    report = gate_report({**metrics(), "sampling": sampling, "samples": 200}, DEFAULTS)
    assert report["passed"] and report["failures"] == []
    assert report["sampling"] == sampling
    assert report["samples"] == 200


def test_gate_matches_recorded_evaluation_artifacts():
    """Guards the thresholds against the real numbers this repo produced."""
    expected = {"finetuned": True, "base": False, "finetuned_maxtok256": False}
    for name, should_pass in expected.items():
        path = Path("artifacts/hdfs/evaluation") / name / "metrics.json"
        if not path.exists():
            continue
        recorded = json.loads(path.read_text(encoding="utf-8"))
        assert run_gate(recorded, DEFAULTS)[0] is should_pass, name
