# Production-oriented automated LLM fine-tuning pipeline

This repository is a practical starting point for automated supervised fine-tuning (SFT) with Axolotl, QLoRA, W&B, Hugging Face Hub, Docker, and GitHub Actions.

## What it does

1. Validates OpenAI-style chat data.
2. Deterministically splits the dataset and records a SHA-256 fingerprint.
3. Renders a run-specific Axolotl configuration.
4. Fine-tunes with QLoRA.
5. Evaluates the candidate model on held-out examples.
6. Fails the pipeline when the configured score gate is not met.
7. Merges the LoRA adapter into a standalone bf16 model.
8. Uploads the candidate to a private Hugging Face model repository when configured.
9. Tags the release immutably and moves the production alias to it.
10. Stores a run manifest and evaluation report with each run.

## Hardware profile

The default settings are intentionally conservative for an 8 GB RTX 4060 Laptop GPU / 16 GB system RAM:

- QLoRA 4-bit loading
- sequence length 2560
- micro batch size 1
- gradient accumulation 8
- gradient checkpointing
- SDPA attention
- no sample packing
- 3 epochs

Increase model size or sequence length only after verifying GPU memory. Larger models should move to a RunPod or dedicated GPU runner.

## Model

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. Change `BASE_MODEL` for another compatible causal language model.

## Local setup

### 1. Prerequisites

Use Linux/WSL2 with a working NVIDIA driver, Docker, and the NVIDIA Container Toolkit.

Check the GPU:

```bash
nvidia-smi
```

Check Docker GPU access:

```bash
make doctor
```

### 2. Configure secrets

```bash
make setup
```

This copies `.env.example` to `.env` (if missing) and creates the artifact/data/cache directories. Edit `.env` and set:

```text
WANDB_API_KEY=...
HF_TOKEN=...
HF_MODEL_REPO=your-org/your-model
```

Never commit `.env` or tokens.

### 3. Validate and test

```bash
make validate
make test
```

### 4. Run locally

```bash
make build
make train
```

The first run downloads the base model. Outputs are stored under `artifacts/<run-id>/`.

## Commands

Generic pipeline:

```bash
make setup      # .env + directories
make validate   # validate data/raw/dataset.jsonl
make prepare    # deterministic split + dataset fingerprint
make test       # pytest -q (same command CI runs)
make build      # docker compose build trainer
make train      # docker compose run --rm trainer
make doctor     # nvidia-smi + Docker GPU sanity check
make logs       # print evaluation.json for every run under artifacts/
make clean      # wipe artifacts/, data/processed/, HF cache
```

Merge a LoRA adapter into a standalone model (host venv, no Docker image needed):

```bash
ADAPTER=<adapter dir> OUTPUT=<output dir> make merge
```

HDFS experiment targets (see `docs/HDFS_ANOMALY_DETECTION.md`):

```bash
make hdfs-dataset-download   # fetch + extract LogHub HDFS_v1 into data/raw/hdfs/
make hdfs-dataset-prepare    # traces + labels -> SFT JSONL, trace-level 80/10/10 split
make hdfs-preflight          # GPU/software checks, fails fast instead of a cryptic OOM
make hdfs-smoke-train        # ~7 min end-to-end smoke run (200 samples)
make hdfs-staged-train       # practical run, HDFS_TRAIN_SAMPLES=20000 default (~10 h)
make hdfs-train              # full 460k dataset -- dedicated GPU only (~10-15 days on 8 GB)
```

Stages are also reachable through the thin dispatcher in `src/cli.py`:

```bash
python -m src.cli dataset download --name hdfs --data-dir data/raw/hdfs
python -m src.cli dataset prepare  --name hdfs --raw-dir data/raw/hdfs --output-dir artifacts/hdfs/processed
python -m src.cli preflight
python -m src.cli train    --config configs/qwen2.5-1.5b-hdfs-qlora.yml
python -m src.cli merge    --adapter <adapter dir> --output <output dir>
python -m src.cli evaluate ...
python -m src.cli compare  ...
python -m src.cli infer    ...
```

It forwards argv to the matching module; invoking modules directly (`python -m src.data.validate ...`) still works and remains the repo convention.

## Dataset format

Use one JSON object per line:

```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"Question"},
  {"role":"assistant","content":"Answer"}
]}
```

For a production project, replace the toy examples in `data/raw/dataset.jsonl` with your curated dataset and put the raw data in a controlled storage system when it becomes too large for Git.

## GitHub Actions production flow

The `train.yml` workflow is manual by design. This avoids accidentally burning GPU hours on every push.

Required repository secrets:

- `HF_TOKEN`
- `HF_MODEL_REPO`
- `WANDB_API_KEY`
- `WANDB_ENTITY` (optional)

The training job expects a self-hosted runner with these labels:

```text
self-hosted, linux, x64, gpu
```

On GitHub: Settings -> Actions -> Runners -> New self-hosted runner.

The runner must be able to run Docker and access the NVIDIA GPU.

## RunPod production pattern

For production training, do not make your laptop the long-lived GPU runner. Use a dedicated or ephemeral RunPod GPU environment and give it the same `gpu` label. Keep the container, config, and GitHub workflow identical so the training workload is portable.

A second production pattern is to build a versioned training image and use RunPod GitHub integration or a Pod created from the RunPod REST API. Keep the image tag tied to the Git commit/release, not `latest`.

## W&B

W&B is the only experiment tracker enabled in this template. It records the training run, hyperparameters, and metrics without adding multiple logging backends.

## Evaluation gate

The generic evaluator (`src/evaluation/evaluate.py`) uses a simple token-level F1 score against `EVAL_MIN_F1`. This is intentionally generic, but it is not sufficient for a serious domain model.

For production, replace it with task-specific evaluation such as:

- classification accuracy/F1
- structured JSON validity
- retrieval-grounded answer accuracy
- tool-call correctness
- safety policy pass rate
- human preference score

The release gate should use metrics that actually reflect your product.

The HDFS experiment is the worked example of that. `src/evaluation/hdfs_gate.py` turns the evaluation metrics into a pass/fail decision on five thresholds, all configurable in `.env`:

| Metric | Env var | Default |
|---|---|---|
| JSON schema validity % | `HDFS_GATE_MIN_SCHEMA_VALID_PCT` | 90.0 |
| Accuracy | `HDFS_GATE_MIN_ACCURACY` | 0.90 |
| Anomaly recall | `HDFS_GATE_MIN_ANOMALY_RECALL` | 0.85 |
| Anomaly F1 | `HDFS_GATE_MIN_ANOMALY_F1` | 0.85 |
| Evidence grounding rate | `HDFS_GATE_MIN_EVIDENCE_GROUNDING` | 0.90 |

Defaults sit roughly 10 points under the observed fine-tuned run, so they absorb run-to-run variance while still catching the two failure modes this repo has actually produced: an unadapted base model, and a truncated-output run caused by too small an `EVAL_MAX_NEW_TOKENS` budget. Recall and F1 are both checked on purpose -- a model that answers "anomaly" every time scores recall 1.0, and only F1 catches it. The run writes `gate.json` next to its evaluation report, and nothing reaches the model registry unless the gate passes.

## Model promotion

A passing evaluation gate is a prerequisite for upload. Each gated release then gets two pointers in the Hugging Face repo (`src/registry/upload_to_hub.py`):

- `run-<run-id>` -- an immutable tag pinning the exact commit for that run.
- `<HF_PRODUCTION_ALIAS>` -- a movable branch (default `production`) that serving pins to via `revision=`.

The tag is created before the alias moves, so a failed alias move still leaves the release addressable. Rollback is pointing the alias at an earlier `run-*` tag. Set `HF_PRODUCTION_ALIAS` empty to upload and tag without promoting.

For a stricter rollout, add a human approval stage before the alias moves, or keep separate `staging` and `production` aliases. Do not deploy every successful training run automatically.

## HDFS anomaly-classification experiment

A concrete experiment built on top of this pipeline: fine-tuning
`Qwen/Qwen2.5-1.5B-Instruct` with QLoRA on the LogHub HDFS_v1 dataset to
classify raw HDFS log traces as `normal`/`anomaly` and return structured
JSON. See `docs/HDFS_ANOMALY_DETECTION.md` for commands, dataset
reconstruction details, config, and evaluation.

## Documentation

- `docs/HDFS_ANOMALY_DETECTION.md` -- the HDFS experiment end to end: commands, artifact layout, output schema, measured training cost, limitations.
- `docs/OPERATIONS.md` -- day-to-day operation: normal development loop, training a new dataset, launching a production run, OOM debugging order, releasing.
- `docs/PRODUCTION.md` -- production checklist: repository protection, runner, container pinning, model registry, deployment, rollback.

## Reproducibility

Every run stores:

- model name and revision
- dataset fingerprint
- random seed
- Git SHA when available
- evaluation report
- training config

For strict enterprise reproducibility, pin Docker image digests and use immutable dataset/model revisions instead of floating `main`.
