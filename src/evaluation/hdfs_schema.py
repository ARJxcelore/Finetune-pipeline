"""Strict parsing/validation of the HDFS incident-classification JSON output."""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

REQUIRED_KEYS = {"is_anomaly", "incident_type", "confidence", "evidence", "summary"}
VALID_INCIDENT_TYPES = {"normal", "anomaly"}

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> str | None:
    """Extract the first/largest-looking `{...}` span from model output.

    Models sometimes wrap JSON in prose or code fences; this is a best-effort,
    deterministic extraction, not a repair tool.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = _JSON_OBJECT_RE.search(text)
    return match.group(0) if match else None


class ParsedResponse(NamedTuple):
    """Result of parsing a model response.

    `json_obj` is set whenever the output parsed as a JSON object at all, even
    if it then failed schema validation -- JSON validity and schema validity
    are separate metrics (a model can emit well-formed JSON with
    `confidence: 95` or `evidence` as a list of objects), and collapsing them
    into one flag hides exactly the failure mode fine-tuning is meant to fix.
    """

    json_obj: dict[str, Any] | None
    schema_valid: bool
    errors: list[str]

    @property
    def json_valid(self) -> bool:
        return self.json_obj is not None

    @property
    def predicted_is_anomaly(self) -> bool | None:
        """The classification decision, if the model expressed one as a boolean.

        Read independently of full schema validity so classification accuracy
        and schema compliance can be reported as the distinct things they are.
        """
        if self.json_obj is None:
            return None
        value = self.json_obj.get("is_anomaly")
        return value if isinstance(value, bool) else None


def parse_and_validate(text: str) -> ParsedResponse:
    """Parse a model response and validate it against the output schema."""
    errors: list[str] = []
    candidate = extract_json_object(text)
    if candidate is None:
        return ParsedResponse(None, False, ["no JSON object found in output"])

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedResponse(None, False, [f"invalid JSON: {exc}"])

    if not isinstance(obj, dict):
        return ParsedResponse(None, False, ["parsed JSON is not an object"])

    missing = REQUIRED_KEYS - obj.keys()
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")

    if "is_anomaly" in obj and not isinstance(obj["is_anomaly"], bool):
        errors.append("is_anomaly must be a boolean")
    if "incident_type" in obj:
        if not isinstance(obj["incident_type"], str):
            errors.append("incident_type must be a string")
        elif obj["incident_type"] not in VALID_INCIDENT_TYPES:
            errors.append(f"incident_type must be one of {sorted(VALID_INCIDENT_TYPES)}")
    if "confidence" in obj:
        if not isinstance(obj["confidence"], (int, float)) or isinstance(obj["confidence"], bool):
            errors.append("confidence must be a number")
        elif not 0 <= obj["confidence"] <= 1:
            errors.append("confidence must be within [0, 1]")
    if "evidence" in obj:
        if not isinstance(obj["evidence"], list) or not all(isinstance(e, str) for e in obj["evidence"]):
            errors.append("evidence must be a list of strings")
    if "summary" in obj and not isinstance(obj["summary"], str):
        errors.append("summary must be a string")

    return ParsedResponse(obj, not errors, errors)


def evidence_grounding_rate(evidence: list[str], source_trace: str) -> float:
    """Fraction of evidence lines that are a verbatim substring of the source trace."""
    if not evidence:
        return 1.0
    grounded = sum(1 for line in evidence if line.strip() and line.strip() in source_trace)
    return grounded / len(evidence)
