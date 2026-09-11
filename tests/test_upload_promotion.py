import json
from types import SimpleNamespace

import pytest

from src.registry.upload_to_hub import move_alias, refuse_if_gate_failed


def write_manifest(tmp_path, payload):
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_failed_gate_blocks_upload(tmp_path):
    with pytest.raises(SystemExit):
        refuse_if_gate_failed(write_manifest(tmp_path, {"gate_passed": False}))


def test_passed_gate_allows_upload(tmp_path):
    refuse_if_gate_failed(write_manifest(tmp_path, {"gate_passed": True}))


def test_manifest_without_gate_key_is_allowed(tmp_path):
    """src.training.pipeline gates upstream, so absence is 'n/a', not 'unchecked'."""
    refuse_if_gate_failed(write_manifest(tmp_path, {"run_id": "x"}))


def test_missing_manifest_is_allowed(tmp_path):
    refuse_if_gate_failed(tmp_path / "does_not_exist.json")


class FakeApi:
    def __init__(self, branches):
        self.branches = list(branches)
        self.calls = []

    def list_repo_refs(self, repo_id, repo_type=None, token=None):
        return SimpleNamespace(branches=[SimpleNamespace(name=b) for b in self.branches])

    def delete_branch(self, repo_id, branch, repo_type=None, token=None):
        self.calls.append(("delete", branch, None))
        self.branches.remove(branch)

    def create_branch(self, repo_id, branch, revision=None, repo_type=None, token=None):
        self.calls.append(("create", branch, revision))
        self.branches.append(branch)


def test_alias_is_created_when_absent():
    api = FakeApi(["main"])
    move_alias(api, "org/model", "production", "abc123", "tok")
    assert api.calls == [("create", "production", "abc123")]


def test_existing_alias_is_deleted_then_recreated_at_new_revision():
    api = FakeApi(["main", "production"])
    move_alias(api, "org/model", "production", "def456", "tok")
    assert api.calls == [("delete", "production", None), ("create", "production", "def456")]
    assert "production" in api.branches
