from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any



def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize(prediction)
    ref = normalize(reference)
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    pred_counts: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for token in pred:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = sum(min(pred_counts.get(tok, 0), count) for tok, count in ref_counts.items())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def last_assistant(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    raise ValueError("No assistant response in evaluation example")


def prompt_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx]
    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=os.environ["BASE_MODEL"])
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-f1", type=float, default=float(os.getenv("EVAL_MIN_F1", "0.35")))
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows: list[dict[str, Any]] = []
    with args.dataset.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip() and len(rows) < args.max_samples:
                rows.append(json.loads(raw))

    results: list[dict[str, Any]] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        messages = row["messages"]
        reference = last_assistant(messages)
        prompt = prompt_messages(messages)
        inputs = tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        prediction = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
        score = token_f1(prediction, reference)
        scores.append(score)
        results.append({
            "index": index,
            "score": score,
            "reference": reference,
            "prediction": prediction,
        })

    average_f1 = sum(scores) / len(scores) if scores else 0.0
    payload = {
        "metric": "token_f1",
        "average_f1": average_f1,
        "min_f1_threshold": args.min_f1,
        "passed": average_f1 >= args.min_f1,
        "samples": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
