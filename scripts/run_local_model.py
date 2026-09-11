from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================
# Configuration
# =========================

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(REPO_ROOT / "artifacts/20260907-123334/axolotl/merged")

PROMPT = """
Ayush Jaiswal
AWS Certified —— AI/ML Engineer
Noida (U.P.) — ayush.jaiswal.at.work@gmail.com — 6299552593 — linkedin.com/in/ayush-raj-jaiswal/
Experience
Senior AI Engineer, Xcelore Private Limited
( Sept 2023 – Present )
• Built end-to-end LLM lifecycle pipelines (SFT, DPO, GPTQ, GGUF) serving production traffic; reduced in-
ference latency by 28% via quantization and optimized batching.
• Designed hybrid RAG architecture (FAISS + Pinecone + cross-encoder reranker) with FastAPI serving layer;
achieved p95 latency of 850ms under concurrent load.
• Engineered voice-to-voice RAG assistant (STT → retrieval → LLM → TTS) deployed on AWS; reduced average
response time by 32% using embedding cache + async tool calls.
• Developed e-commerce recommendation service (FastAPI + Elasticsearch + Redis + ONNX) improving re-
call@10 by 35% and reducing search latency by 59% using HNSW indexing.
• Deployed 7+ ML services on AWS/GCP using Docker and CI/CD; reduced infrastructure cost by 30% through
GPU right-sizing and autoscaling.
• Implemented monitoring (request latency, token usage, error rate) and added retries + fallback models for
fail-open reliability.
• Awarded Impactful Xcelorean Award for delivering high-impact AI systems that increased automation effi-
ciency and accelerated enterprise AI adoption.
Software Engineer, BNP Paribas India Solutions
( Jan 2023 – June 2023 )
• Automated CI/CD pipelines (GitLab, Jenkins, JFrog, Ansible) reducing deployment time by 40%.
• Managed containerized deployments with Docker; improved release stability and reduced production incidents.
Projects
Concurrent LLM Streaming Service with vLLM
• Architecture: FastAPI → vLLM → Continuous Batching → PagedAttention → Streaming API
• Built a concurrent streaming LLM service handling 20 simultaneous requests on RTX 4060 (8GB VRAM).
• Leveraged token-level continuous batching to eliminate batch window delay and reduce tail latency.
• Tuned concurrency (max-num-seqs, max-num-batched-tokens) (40t/s prompt; 160t/s gen; 67% cache hit).
• Delivered stable multi-user performance without request-queue bottlenecks on a single GPU.
• Built monitoring (latency/token/error) and retries/fallbacks for fail-open reliability.
• Tech: vLLM, SmolLM2-1.7B-Instruct, FastAPI, Docker, CUDA
Invoice Information Extraction System
• Built OCR + layout-aware transformer pipeline achieving F1-score 0.89.
• Deployed REST API with p95 latency 720ms per document.
Technical Skills
Machine Learning & NLP: PyTorch, TensorFlow, Scikit-Learn, Transformers, LangChain, Spacy, XGBoost, OpenCV,
FAISS, Hugging Face, OpenAI, Groq
Backend & MLOps: FastAPI, Flask, Docker, AWS, GCP, Lightning Serve, Git
Databases & Data Stores: MySQL, MongoDB, Elasticsearch, Pinecone, Qdrant, Redis, Neo4j
Languages & Visualization: Python, Power BI, Tableau, Jupyter, Google Colab
Publications and Certifications
• Oversampled Deep Fully Connected Neural Network Towards Improving Classifier Performance for
Fraud Detection
( Nov 2022 )
• Impactful Xcelorean Award, Xcelore Pvt. Ltd
( Jan 2025 )
• AWS Certified AI Practitioner, AWS
( May 2025 )
Education
Vel Tech R & D Institute of Science and Technology, B.Tech in Computer Science
( June 2023 )
"""

SYSTEM_PROMPT = """
You are a resume parsing assistant.
Extract information accurately and return structured JSON.
"""

MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
    )

    model.eval()

    messages = []

    if SYSTEM_PROMPT:
        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT,
        })

    messages.append({
        "role": "user",
        "content": PROMPT,
    })

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(next(model.parameters()).device)

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": TEMPERATURE > 0,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **gen_kwargs,
        )

    text = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print(text.strip())


if __name__ == "__main__":
    main()