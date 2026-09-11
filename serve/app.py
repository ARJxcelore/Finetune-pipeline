
from __future__ import annotations

import os
from functools import lru_cache

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID")
HF_TOKEN = os.getenv("HF_TOKEN") or None
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "256"))

app = FastAPI(title="Fine-tuned LLM Service", version="1.0.0")


class GenerateRequest(BaseModel):
    messages: list[dict[str, str]] = Field(min_length=1)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS_DEFAULT, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_ID:
        raise RuntimeError("MODEL_ID is not configured")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN, use_fast=True)
    kwargs = {"token": HF_TOKEN, "device_map": "auto"}
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return tokenizer, model


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/generate")
def generate(request: GenerateRequest):
    try:
        tokenizer, model = load_model()
        inputs = tokenizer.apply_chat_template(
            request.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        kwargs = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if request.temperature > 0:
            kwargs["temperature"] = request.temperature
        with torch.no_grad():
            output = model.generate(inputs, **kwargs)
        text = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return {"model": MODEL_ID, "text": text.strip()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
