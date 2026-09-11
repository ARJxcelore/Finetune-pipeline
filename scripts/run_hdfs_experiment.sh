#!/usr/bin/env bash
#
# One-shot runner for the HDFS anomaly-classification experiment:
#   dataset -> preflight -> build image -> smoke train -> real train
#   -> baseline eval -> fine-tuned eval -> compare -> inference demo
#
# Stops at the first failure, so a broken smoke test never rolls on into a
# multi-hour training run. Every stage is skipped if its output already
# exists, so re-running after a failure resumes instead of starting over.
#
# Usage:
#   ./scripts/run_hdfs_experiment.sh                 # smoke test + 20k-example run (~10 h)
#   TRAIN_SAMPLES=5000 ./scripts/run_hdfs_experiment.sh    # ~2.6 h
#   TRAIN_SAMPLES=0 ./scripts/run_hdfs_experiment.sh       # full 460k dataset (~10-15 days)
#   SKIP_SMOKE=1 ./scripts/run_hdfs_experiment.sh          # skip the smoke test
#   STAGE=eval ./scripts/run_hdfs_experiment.sh            # only re-run eval/compare
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"

# CRITICAL: `docker compose run` FAILS (exit 1, zero output) whenever its stdout is
# a regular FILE rather than a pipe or TTY -- reproduced on docker 29.6.1 (snap),
# with and without -T, redirecting to both /tmp and the repo. Running this script as
# `nohup ./script.sh > run.log` makes the script's stdout a file, every compose call
# inherits it and dies instantly WHILE LEAVING ITS CONTAINER RUNNING. That is what
# produced a 9-hour training run that completed successfully but orphaned, with the
# script reporting failure seconds in and the eval stages never firing.
# Fix: every compose invocation pipes through `cat` so compose sees a pipe. Do not
# "simplify" these pipes away. `-T` is kept because there is no TTY to allocate.

# Single-instance lock. Two concurrent runs split the 7.6 GiB GPU (~2 GiB each when
# ~6.7 is needed) and BOTH die with CUDA OOM -- which is exactly what happened when
# this was launched twice by accident, and it looked like a config failure.
LOCK_FILE="${TMPDIR:-/tmp}/hdfs_experiment.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "XX  Another run of this script is already in progress (lock: $LOCK_FILE)." >&2
  echo "    Two concurrent runs will OOM the GPU. Wait for it, or kill it first." >&2
  exit 1
fi
echo $$ >&9

# ---------------------------------------------------------------- knobs -----
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"   # 0 = full dataset
SMOKE_SAMPLES="${SMOKE_SAMPLES:-200}"
# The natural prior is 2.93% anomaly. Training on that unweighted teaches the model
# that "normal" is right 97% of the time and anomaly recall collapses, which would
# make an overnight run produce a useless model. Set to "" to keep the raw prior.
TRAIN_ANOMALY_RATIO="${TRAIN_ANOMALY_RATIO:-0.3}"
EVAL_ANOMALIES="${EVAL_ANOMALIES:-100}"   # stratified: measures anomaly recall meaningfully
EVAL_NORMALS="${EVAL_NORMALS:-100}"
# Ground-truth anomaly targets need median 275 / p95 316 tokens (they copy up to 5
# verbatim log lines as evidence). 256 truncated over half of them mid-JSON, which
# scored as "invalid JSON" and silently halved measured anomaly recall.
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
CONFIG="${CONFIG:-configs/qwen2.5-1.5b-hdfs-qlora.yml}"
PROCESSED="artifacts/hdfs/processed"
EVAL_DIR="artifacts/hdfs/evaluation"
STAGE="${STAGE:-all}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX  %s\033[0m\n' "$*" >&2; exit 1; }

want_stage() { [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# Every container invocation goes through here. The `| cat` is load-bearing -- see
# the CRITICAL note at the top of this file. pipefail keeps compose's exit status.
compose_py() { docker compose run --rm -T --entrypoint python trainer "$@" 2>&1 | cat; }

trap 'die "failed at line $LINENO -- nothing after this point ran"' ERR

# --------------------------------------------------------------- checks -----
log "Preflight checks"
command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose plugin not available"
[[ -f .env ]] || { warn ".env missing, creating from .env.example"; cp .env.example .env; }

# Works whether or not the venv is active (Ubuntu has no bare `python` system-wide).
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x .venv/bin/python ]]; then PY=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then PY="python"
  else PY="python3"; fi
fi
echo "using interpreter: $PY ($("$PY" --version 2>&1))"

# STOP_EXISTING=1 clears a previous training run of THIS project before starting.
# Scoped deliberately to this compose project's trainer containers -- it will never
# touch an unrelated GPU process, so it can't silently kill someone else's job.
if [[ "${STOP_EXISTING:-0}" == "1" ]]; then
  log "STOP_EXISTING=1: stopping previous trainer containers from this project"
  mapfile -t OLD_TRAINERS < <(docker ps -q --filter "name=fine-tuning-pipeline-prod-trainer" || true)
  if [[ ${#OLD_TRAINERS[@]} -gt 0 ]]; then
    docker ps --filter "name=fine-tuning-pipeline-prod-trainer" \
      --format '  stopping {{.ID}}  {{.Names}}  (up {{.Status}})'
    docker stop "${OLD_TRAINERS[@]}" >/dev/null
    echo "  stopped ${#OLD_TRAINERS[@]} container(s); waiting for VRAM to be released"
    sleep 10
  else
    echo "  none running"
  fi
fi

# The dataset stage is pure CPU, so don't block it on a busy GPU.
if [[ "$STAGE" == "dataset" ]]; then
  GPU_CHECK=0
else
  GPU_CHECK=1
fi

if [[ "$GPU_CHECK" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | grep -q .; then
    # Hard stop, not a warning: this run needs ~6.7 of 7.6 GiB. Starting anyway
    # would OOM partway through an unattended overnight run, which is worse than
    # failing in the first 5 seconds while someone is still watching.
    printf '\033[1;31m'
    echo "XX  The GPU is already in use -- refusing to start (it would OOM):"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo
    echo "    If it is a previous run of this project, re-run with:  STOP_EXISTING=1 $0"
    echo "    Otherwise stop it yourself, then re-run. FORCE_GPU=1 overrides this check."
    printf '\033[0m'
    [[ "${FORCE_GPU:-0}" == "1" ]] || exit 1
  fi
else
  warn "nvidia-smi not found -- cannot check whether the GPU is free"
fi

# -------------------------------------------------------------- dataset -----
if want_stage all || want_stage dataset; then
  log "Stage 1/7: dataset download (LogHub HDFS_v1 via Hugging Face)"
  if [[ -f "data/raw/hdfs/extracted/HDFS.log" && -f "data/raw/hdfs/extracted/anomaly_label.csv" ]]; then
    echo "already present, skipping"
  else
    "$PY" -m src.cli dataset download --name hdfs --data-dir data/raw/hdfs
  fi

  log "Stage 2/7: build SFT dataset (trace reconstruction, 80/10/10 trace-level split)"
  if [[ -f "$PROCESSED/train.jsonl" && -f "$PROCESSED/validation_quick.jsonl" ]]; then
    echo "already built, skipping (delete $PROCESSED to rebuild)"
  else
    # --no-tokenizer-stats skips a ~7 min single-threaded pass over 575k rows;
    # drop it if you want token-length percentiles in dataset_stats.json.
    "$PY" -m src.cli dataset prepare --name hdfs \
      --raw-dir data/raw/hdfs --output-dir "$PROCESSED" --no-tokenizer-stats
  fi

  log "Dataset validation (schema of the generated SFT records)"
  "$PY" -m src.data.validate "$PROCESSED/validation_quick.jsonl"
fi

[[ -f "$PROCESSED/train.jsonl" ]] || die "$PROCESSED/train.jsonl missing -- run with STAGE=dataset first"

# ------------------------------------------------------------- preflight ----
if want_stage all || want_stage train; then
  log "Stage 3/7: GPU/software preflight"
  "$PY" -m src.cli preflight --base-model "$BASE_MODEL" || \
    warn "preflight reported problems (expected on the host: axolotl/bitsandbytes live in the container)"

  log "Stage 4/7: build the training image"
  docker compose build trainer

  # ---------------------------------------------------------- smoke test ----
  if [[ "$SKIP_SMOKE" != "1" ]]; then
    log "Stage 5/7: smoke train ($SMOKE_SAMPLES examples, ~7 min) -- gates the real run"
    SMOKE_RUN_ID="smoke-$(date -u +%Y%m%d-%H%M%S)"
    RUN_ID="$SMOKE_RUN_ID" compose_py \
      -m src.training.hdfs_pipeline --config "$CONFIG" --max-train-samples "$SMOKE_SAMPLES"
    [[ -d "artifacts/hdfs/runs/$SMOKE_RUN_ID" ]] || die "smoke run produced no artifacts"
    log "Smoke test PASSED -> artifacts/hdfs/runs/$SMOKE_RUN_ID"
  else
    warn "SKIP_SMOKE=1 -- skipping the smoke test"
  fi

  # ----------------------------------------------------------- real run ----
  RUN_ID="${RUN_ID:-hdfs-$(date -u +%Y%m%d-%H%M%S)}"
  BALANCE_ARGS=()
  [[ -n "$TRAIN_ANOMALY_RATIO" ]] && BALANCE_ARGS=(--train-anomaly-ratio "$TRAIN_ANOMALY_RATIO")

  if [[ "$TRAIN_SAMPLES" == "0" ]]; then
    log "Stage 6/7: FULL dataset training (~10-15 days on one 8 GB GPU) -- run id $RUN_ID"
    warn "This is a multi-day run. Ctrl+C now and set TRAIN_SAMPLES=20000 if that wasn't intended."
    sleep 15
    RUN_ID="$RUN_ID" compose_py \
      -m src.training.hdfs_pipeline --config "$CONFIG"
  else
    log "Stage 6/7: training on $TRAIN_SAMPLES examples (anomaly ratio ${TRAIN_ANOMALY_RATIO:-natural}) -- run id $RUN_ID"
    RUN_ID="$RUN_ID" compose_py \
      -m src.training.hdfs_pipeline --config "$CONFIG" \
      --max-train-samples "$TRAIN_SAMPLES" "${BALANCE_ARGS[@]}"
  fi
  echo "$RUN_ID" > artifacts/hdfs/.last_run_id
fi

# ------------------------------------------------------------ evaluation ----
if want_stage all || want_stage eval; then
  RUN_ID="${RUN_ID:-$(cat artifacts/hdfs/.last_run_id 2>/dev/null || true)}"
  [[ -n "$RUN_ID" ]] || die "no run id known -- set RUN_ID=<run-id> explicitly"
  RUN_DIR="artifacts/hdfs/runs/$RUN_ID"
  [[ -d "$RUN_DIR" ]] || die "$RUN_DIR not found"

  # Mirror hdfs_pipeline.py's adapter lookup: root output dir, else newest checkpoint-*.
  if [[ -f "$RUN_DIR/checkpoints/adapter_config.json" ]]; then
    ADAPTER="$RUN_DIR/checkpoints"
  else
    # `|| true`: under `set -o pipefail` a failing find (missing dir) would abort the
    # whole script here instead of reaching the clear error message below.
    ADAPTER="$(find "$RUN_DIR/checkpoints" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null | sort -V | tail -1 || true)"
  fi
  [[ -n "$ADAPTER" && -d "$ADAPTER" ]] || die "no LoRA adapter found under $RUN_DIR/checkpoints"
  log "Using adapter: $ADAPTER"

  EVAL_ARGS=(--dataset "$PROCESSED/test.jsonl"
             --stratified-anomalies "$EVAL_ANOMALIES"
             --stratified-normals "$EVAL_NORMALS"
             --max-new-tokens "$EVAL_MAX_NEW_TOKENS")

  # Training runs as root inside the container and writes adapter_model.safetensors
  # mode 600 root:root, so the host user cannot read it -- a host-side eval dies with
  # FileNotFoundError on a file that plainly exists. Run eval in the container (same
  # env as training, guaranteed read access) whenever the adapter isn't host-readable.
  if head -c 1 "$ADAPTER/adapter_model.safetensors" >/dev/null 2>&1; then
    EVAL_IN_CONTAINER=0
    echo "adapter is host-readable; evaluating on the host"
  else
    EVAL_IN_CONTAINER=1
    warn "adapter is root-owned (written by the training container); evaluating inside the container"
  fi
  eval_py() {
    if [[ "$EVAL_IN_CONTAINER" == "1" ]]; then compose_py -m "$@"; else "$PY" -m "$@"; fi
  }

  log "Stage 7a/7: BASELINE eval (base model, no adapter) on the held-out test set"
  eval_py src.cli evaluate --base-model "$BASE_MODEL" \
    "${EVAL_ARGS[@]}" --output-dir "$EVAL_DIR/base"

  log "Stage 7b/7: FINE-TUNED eval (same examples, same prompt, same settings)"
  eval_py src.cli evaluate --base-model "$BASE_MODEL" --adapter "$ADAPTER" \
    "${EVAL_ARGS[@]}" --output-dir "$EVAL_DIR/finetuned"

  log "Stage 7c/7: base vs fine-tuned comparison + qualitative report"
  eval_py src.cli compare \
    --base-predictions "$EVAL_DIR/base/predictions.jsonl" \
    --finetuned-predictions "$EVAL_DIR/finetuned/predictions.jsonl" \
    --output-dir "$EVAL_DIR"

  log "Inference demo on a single trace"
  eval_py src.cli infer --base-model "$BASE_MODEL" --model-path "$ADAPTER" \
    --input-file examples/hdfs_trace.txt || warn "inference demo failed (non-fatal)"
fi

# ----------------------------------------------------------------- done -----
log "DONE"
cat <<EOF

Results:
  dataset stats        $PROCESSED/dataset_stats.json
  run artifacts        artifacts/hdfs/runs/${RUN_ID:-<run-id>}/
  run metadata         artifacts/hdfs/runs/${RUN_ID:-<run-id>}/run_metadata.json
  baseline metrics     $EVAL_DIR/base/metrics.json
  fine-tuned metrics   $EVAL_DIR/finetuned/metrics.json
  side-by-side         $EVAL_DIR/comparison.json
  qualitative report   $EVAL_DIR/qualitative_report.md

Read the "sampling" block in metrics.json before quoting precision: the eval
uses a stratified sample so anomaly recall is measurable, which makes precision
and accuracy optimistic relative to the true 2.93% anomaly prior.
EOF
