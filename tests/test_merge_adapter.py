import json

import pytest

from src.training.merge_adapter import copy_tokenizer_files, resolve_base_model


def _write_adapter(tmp_path, config):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return adapter


def test_resolve_base_model_from_adapter_config(tmp_path):
    adapter = _write_adapter(tmp_path, {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert resolve_base_model(adapter, None) == "Qwen/Qwen2.5-1.5B-Instruct"


def test_explicit_base_model_wins(tmp_path):
    adapter = _write_adapter(tmp_path, {"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert resolve_base_model(adapter, "other/model") == "other/model"


def test_missing_adapter_config_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        resolve_base_model(empty, None)


def test_missing_base_model_field_raises(tmp_path):
    adapter = _write_adapter(tmp_path, {"r": 16})
    with pytest.raises(SystemExit):
        resolve_base_model(adapter, None)


def test_copy_tokenizer_files_copies_only_present(tmp_path):
    adapter = _write_adapter(tmp_path, {"base_model_name_or_path": "x"})
    (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")
    (adapter / "chat_template.jinja").write_text("tpl", encoding="utf-8")
    output = tmp_path / "merged"
    output.mkdir()

    copied = copy_tokenizer_files(adapter, output)

    assert sorted(copied) == ["chat_template.jinja", "tokenizer.json"]
    assert (output / "chat_template.jinja").read_text(encoding="utf-8") == "tpl"
