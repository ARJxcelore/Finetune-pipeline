"""Shared model loading + generation for HDFS baseline eval, fine-tuned eval, and inference.

Kept in one place so the evaluation prompt/preprocessing is guaranteed identical
across the base model, the fine-tuned model, and single-trace inference.
"""

from __future__ import annotations

from typing import Any


def load_model(base_model: str, adapter_path: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def generate(tokenizer: Any, model: Any, prompt_messages: list[dict[str, str]], max_new_tokens: int = 256) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()


def last_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError("No assistant response in example")


def prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx]
    return messages


def source_trace(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""
