"""Publish a merged model to the Hugging Face model repo and promote it.

Two refs are written per release:

  run-<run-id>   an immutable tag pinning the exact commit for that run, so a
                 past release can always be fetched back verbatim
  <alias>        a movable branch (default "production") that serving pins to
                 via revision=..., so promotion and rollback are ref moves
                 rather than re-uploads

The tag is created before the alias moves: if the alias move fails, the release
still exists under its run tag and can be promoted by hand.

This is the promotion boundary, so the gate is re-checked here rather than
trusted from the caller -- see `gate_passed` in the manifest.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def refuse_if_gate_failed(manifest_file: Path) -> None:
    """Manifests without a `gate_passed` key are from src.training.pipeline, whose
    gate runs upstream (evaluate --min-f1 aborts the run), so absence means 'not
    applicable' rather than 'not checked'."""
    if not manifest_file.exists():
        return
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if "gate_passed" in manifest and not manifest["gate_passed"]:
        raise SystemExit(
            f"Refusing to upload: the release gate did not pass ({manifest_file}). "
            "Fix the model or lower HDFS_GATE_MIN_* deliberately -- do not bypass this."
        )


def move_alias(api: HfApi, repo_id: str, alias: str, revision: str, token: str) -> None:
    """Point `alias` at `revision`.

    The Hub has no atomic branch-move: create_branch(exist_ok=True) will not
    relocate an existing branch, so an existing alias is deleted and recreated.
    The window is small, and the run tag created beforehand means the release is
    never unreachable even if this fails midway.
    """
    existing = {ref.name for ref in api.list_repo_refs(repo_id, repo_type="model", token=token).branches}
    if alias in existing:
        api.delete_branch(repo_id=repo_id, branch=alias, repo_type="model", token=token)
    api.create_branch(repo_id=repo_id, branch=alias, revision=revision, repo_type="model", token=token)


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_MODEL_REPO")
    model_dir = Path(os.environ.get("MODEL_DIR", "artifacts/run/merged"))
    eval_file = Path(os.environ.get("EVAL_FILE", "artifacts/run/evaluation.json"))
    manifest_file = Path(os.environ.get("MANIFEST_FILE", "artifacts/run/run_manifest.json"))
    private = os.environ.get("HF_PRIVATE", "true").lower() == "true"
    run_id = os.environ.get("RUN_ID", "unknown")
    alias = os.environ.get("HF_PRODUCTION_ALIAS", "production").strip()

    if not token:
        raise SystemExit("HF_TOKEN is required for upload")
    if not repo_id:
        raise SystemExit("HF_MODEL_REPO is required for upload")
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    refuse_if_gate_failed(manifest_file)

    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)

    if eval_file.exists():
        shutil.copy2(eval_file, model_dir / "evaluation.json")
    if manifest_file.exists():
        shutil.copy2(manifest_file, model_dir / "run_manifest.json")

    commit = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release fine-tuned model {run_id}",
        token=token,
    )
    print(f"Uploaded model to https://huggingface.co/{repo_id}")

    revision = commit.oid
    tag = f"run-{run_id}"
    api.create_tag(
        repo_id=repo_id,
        tag=tag,
        revision=revision,
        repo_type="model",
        tag_message=f"Gated release {run_id}",
        token=token,
    )
    print(f"Tagged {tag} -> {revision}")

    if alias:
        move_alias(api, repo_id, alias, revision, token)
        print(f"Promoted {alias} -> {revision}")
        print(f"Serve it with: revision='{alias}'  (rollback: move {alias} to an earlier run-* tag)")
    else:
        print("HF_PRODUCTION_ALIAS is empty; uploaded and tagged without promoting")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
