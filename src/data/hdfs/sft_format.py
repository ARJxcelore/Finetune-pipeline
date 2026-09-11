"""Convert an HDFSTrace into the SFT chat record used for this experiment.

Output schema (identical for every example, ground truth is HDFS_v1's
block-level normal/anomaly label only -- this is anomaly classification,
not root-cause analysis):

    {
      "is_anomaly": bool,
      "incident_type": "normal" | "anomaly",
      "confidence": float,   # 1.0 for ground truth
      "evidence": list[str], # empty for normal traces
      "summary": str
    }
"""

from __future__ import annotations

import json
from typing import Any

from src.data.hdfs.schema import HDFSTrace

SYSTEM_PROMPT = (
    "You are an HDFS incident analysis assistant. Analyze the supplied HDFS log trace "
    "and return only valid JSON with keys is_anomaly, incident_type, confidence, evidence, "
    "summary. incident_type must be exactly \"normal\" or \"anomaly\" -- this dataset does "
    "not provide a more specific root cause."
)

ANOMALY_SIGNAL_RE_PARTS = ("error", "exception", "fail", "warn", "terminating", "interrupt", "corrupt")
MAX_EVIDENCE_LINES = 5


def render_user_content(raw_logs: list[str]) -> str:
    return "\n".join(raw_logs)


def select_evidence(raw_logs: list[str], max_lines: int = MAX_EVIDENCE_LINES) -> list[str]:
    """Pick the most informative lines from an anomalous trace without inventing content.

    Preference order: lines containing an anomaly-signal keyword (ERROR/WARN/
    exception/etc, case-insensitive), in original order; if fewer than
    max_lines are found this way, pad with the trace's last lines (often the
    terminal/most-recent state of the block).
    """
    signal_lines = [
        line for line in raw_logs if any(part in line.lower() for part in ANOMALY_SIGNAL_RE_PARTS)
    ]
    evidence = signal_lines[:max_lines]
    if len(evidence) < max_lines:
        for line in reversed(raw_logs):
            if len(evidence) >= max_lines:
                break
            if line not in evidence:
                evidence.append(line)
    # Preserve original trace order in the final evidence list.
    order = {line: idx for idx, line in enumerate(raw_logs)}
    evidence.sort(key=lambda line: order.get(line, 0))
    return evidence[:max_lines]


def build_target(trace: HDFSTrace) -> dict[str, Any]:
    if trace.label == "anomaly":
        evidence = select_evidence(trace.raw_logs)
        return {
            "is_anomaly": True,
            "incident_type": "anomaly",
            "confidence": 1.0,
            "evidence": evidence,
            "summary": "The HDFS trace exhibits anomalous behavior.",
        }
    return {
        "is_anomaly": False,
        "incident_type": "normal",
        "confidence": 1.0,
        "evidence": [],
        "summary": "The HDFS trace is classified as normal.",
    }


def build_sft_record(trace: HDFSTrace) -> dict[str, Any]:
    target = build_target(trace)
    assistant_content = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return {
        "trace_id": trace.trace_id,
        "label": trace.label,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_user_content(trace.raw_logs)},
            {"role": "assistant", "content": assistant_content},
        ],
    }
