SHELL := /bin/bash

setup:
	cp -n .env.example .env || true
	mkdir -p artifacts data/processed .cache/huggingface

validate:
	python -m src.data.validate data/raw/dataset.jsonl

prepare:
	python -m src.data.prepare --input data/raw/dataset.jsonl --output-dir data/processed --valid-ratio $${VALID_RATIO:-0.2} --seed $${SEED:-42}

build:
	docker compose build trainer

doctor:
	nvidia-smi
	docker run --rm --gpus all --runtime=nvidia nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi

train:
	docker compose run --rm trainer

logs:
	find artifacts -maxdepth 2 -name evaluation.json -print -exec cat {} \;

clean:
	rm -rf artifacts/* data/processed/* .cache/huggingface
	mkdir -p artifacts data/processed .cache/huggingface

test:
	python -m pytest -q

# --- HDFS anomaly-classification experiment (docs/HDFS_ANOMALY_DETECTION.md) ---

hdfs-dataset-download:
	python -m src.cli dataset download --name hdfs --data-dir data/raw/hdfs

hdfs-dataset-prepare:
	python -m src.cli dataset prepare --name hdfs --raw-dir data/raw/hdfs --output-dir artifacts/hdfs/processed

hdfs-preflight:
	python -m src.cli preflight

hdfs-smoke-train:
	docker compose run --rm --entrypoint python trainer -m src.training.hdfs_pipeline --config configs/qwen2.5-1.5b-hdfs-qlora.yml --max-train-samples 200

# ~10 h on an 8 GB laptop GPU -- the practical "real" run. Override the size with
# HDFS_TRAIN_SAMPLES=5000 make hdfs-staged-train (~2.6 h).
hdfs-staged-train:
	docker compose run --rm --entrypoint python trainer -m src.training.hdfs_pipeline --config configs/qwen2.5-1.5b-hdfs-qlora.yml --max-train-samples $${HDFS_TRAIN_SAMPLES:-20000}

# Full 460k dataset: ~10-15 days on one 8 GB GPU. Use a dedicated GPU (see runpod/).
hdfs-train:
	docker compose run --rm --entrypoint python trainer -m src.training.hdfs_pipeline --config configs/qwen2.5-1.5b-hdfs-qlora.yml

# Merge a LoRA adapter into a standalone model. Runs on the host venv (no Docker
# image needed): ADAPTER=<adapter dir> OUTPUT=<output dir> make merge
merge:
	python -m src.cli merge --adapter $${ADAPTER:?set ADAPTER=<adapter dir>} --output $${OUTPUT:?set OUTPUT=<output dir>}
