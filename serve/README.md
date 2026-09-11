# Serving the released model

Build and run the standalone inference image after a model has passed evaluation.

```bash
cd serve
export MODEL_ID=your-org/your-private-model
export HF_TOKEN=...
docker compose up --build
```

Health check:

```bash
curl http://localhost:8080/health
```

Generate:

```bash
curl -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is fine-tuning?"}]}'
```

For higher-throughput production serving, use the model server recommended for your model family (for example vLLM/SGLang) instead of this simple reference FastAPI server.
