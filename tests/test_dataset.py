import json
from pathlib import Path

from src.data.validate import validate_file


def test_sample_dataset_validates():
    count, errors = validate_file(Path("data/raw/dataset.jsonl"))
    assert count > 0
    assert errors == ""


def test_messages_shape():
    first = json.loads(Path("data/raw/dataset.jsonl").read_text().splitlines()[0])
    assert isinstance(first["messages"], list)
    assert any(m["role"] == "assistant" for m in first["messages"])
