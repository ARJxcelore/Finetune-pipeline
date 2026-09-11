# HDFS log anomaly classification

A fine-tuning experiment built on top of this repository's existing
validate -> prepare -> render -> train -> evaluate -> merge -> upload
pipeline (see root `README.md`). It reuses `src/training/pipeline.py`'s
conventions but is a separate, self-contained experiment under `artifacts/hdfs/`
so it never mixes with the generic toy-dataset pipeline.

## What this model does

```
Raw HDFS log trace (one HDFS block's log lines, in order)
      v
Qwen2.5-1.5B-Instruct + QLoRA (4-bit NF4)
      v
Structured JSON: is_anomaly, incident_type, confidence, evidence, summary
```

## What it does NOT do

This is **anomaly/incident classification**, not root-cause analysis.
LogHub HDFS_v1 provides only a block-level `normal`/`anomaly` ground-truth
label, produced by handcrafted rules -- it does not label *why* a block is
anomalous. Accordingly:

- `incident_type` is restricted to `"normal"` / `"anomaly"`. The model is
  never trained to claim a more specific failure category (e.g. "disk
  failure", "network partition") because no such ground truth exists here.
- `evidence` is a short list of verbatim log lines from the trace, not an
  inferred root cause.
- Do not market or deploy this as a general-purpose HDFS RCA model without a
  dataset that actually carries root-cause labels.

## Data source

Official LogHub HDFS_v1, downloaded from the link published in the loghub
repository (https://github.com/logpai/loghub):
`https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1`
(1,175,629 raw log lines / ~1.47 GiB uncompressed, 575k+ labeled blocks).
The archive is downloaded and extracted read-only; nothing under
`data/raw/hdfs/` is ever modified in place.

Traces are reconstructed **from the raw log text** (`HDFS.log`), grouped by
the `blk_<id>` token every line carries, not from the event-id-only helper
files (`Event_traces.csv`, `HDFS_templates.csv`). This is deliberate: the
model must see actual log text (`INFO dfs.DataNode$PacketResponder ...`),
not an opaque event sequence (`E1 E2 E3`). `anomaly_label.csv` supplies the
per-block ground truth, joined onto the reconstructed traces by block id.

## Commands

```bash
# 1. Download + extract the official archive into data/raw/hdfs/
python -m src.cli dataset download --name hdfs --data-dir data/raw/hdfs

# 2. Reconstruct traces, join labels, build SFT JSONL, trace-level 80/10/10 split,
#    write artifacts/hdfs/processed/{train,validation,test}.jsonl + dataset_stats.json
python -m src.cli dataset prepare --name hdfs \
  --raw-dir data/raw/hdfs --output-dir artifacts/hdfs/processed

# 3. GPU/software preflight (fails fast instead of a cryptic CUDA OOM)
python -m src.cli preflight --base-model Qwen/Qwen2.5-1.5B-Instruct

# 4. Smoke test (~7 min: dataset load, tokenization, model load, LoRA, fwd/bwd, checkpoint)
python -m src.cli train --config configs/qwen2.5-1.5b-hdfs-qlora.yml --max-train-samples 200

# 5. Staged real run -- see the measured-cost table below before choosing a scale.
#    20k/2 epochs is ~10 h on an 8 GB laptop GPU; the full 460k dataset is ~10-15 days
#    and belongs on RunPod, not a laptop.
python -m src.cli train --config configs/qwen2.5-1.5b-hdfs-qlora.yml --max-train-samples 20000

# 5b. Full dataset (only on a dedicated/bigger GPU -- see runpod/)
python -m src.cli train --config configs/qwen2.5-1.5b-hdfs-qlora.yml

# NOTE: steps 6-7 are only needed for a base-vs-fine-tuned comparison. The
# training pipeline already evaluates the merged model and applies the release
# gate itself (artifacts/hdfs/runs/<run-id>/evaluation/ and gate.json), and
# uploads to HF_MODEL_REPO only when that gate passes.

# 6. Baseline evaluation (base model, no adapter) on the held-out test set
python -m src.cli evaluate \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset artifacts/hdfs/processed/test.jsonl \
  --output-dir artifacts/hdfs/evaluation/base

# 7. Fine-tuned evaluation on the same test set
python -m src.cli evaluate \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter artifacts/hdfs/runs/<run-id>/checkpoints \
  --dataset artifacts/hdfs/processed/test.jsonl \
  --output-dir artifacts/hdfs/evaluation/finetuned

# 8. Compare base vs fine-tuned, build the qualitative report
python -m src.cli compare \
  --base-predictions artifacts/hdfs/evaluation/base/predictions.jsonl \
  --finetuned-predictions artifacts/hdfs/evaluation/finetuned/predictions.jsonl \
  --output-dir artifacts/hdfs/evaluation

# 9. Inference on a single trace
python -m src.cli infer \
  --model-path artifacts/hdfs/runs/<run-id>/final \
  --input-file examples/hdfs_trace.txt
```

`python -m src.cli ...` is a thin router onto the modules above (e.g. `dataset
prepare` calls `src.data.hdfs.build_dataset`); every command also works as a
direct module invocation (`python -m src.data.hdfs.build_dataset ...`),
matching this repo's existing convention.

`make hdfs-dataset-download`, `make hdfs-dataset-prepare`, `make hdfs-preflight`,
`make hdfs-smoke-train`, and `make hdfs-train` wrap the commands above.

### Running training steps inside the pinned Axolotl container

Dataset prep, evaluation, and inference need only this repo's `requirements.txt`
and can run on the host. `train`'s Axolotl steps need the pinned
`axolotlai/axolotl:0.18.0` image. `docker-compose.yml`'s `trainer` service has a
fixed entrypoint (`scripts/run_pipeline.sh` -> the generic `src.training.pipeline`),
so run the HDFS pipeline by overriding the entrypoint instead:

```bash
make build   # build the trainer image once
docker compose run --rm --entrypoint python trainer -m src.training.hdfs_pipeline \
  --config configs/qwen2.5-1.5b-hdfs-qlora.yml --max-train-samples 200
```

Drop `--max-train-samples` for the full run. Requires `BASE_MODEL`,
`WANDB_API_KEY`, etc. set in `.env` as usual.

## Artifact layout

```
artifacts/hdfs/
  processed/
    train.jsonl
    validation.jsonl
    test.jsonl
    dataset_stats.json
    dataset_manifest.json
  runs/<run-id>/
    preflight.json
    train.yml            # rendered Axolotl config
    checkpoints/          # Axolotl output_dir (adapter + merged/)
    final -> checkpoints/merged   # symlink used by evaluate/infer
    evaluation/{metrics.json,predictions.jsonl}   # in-pipeline eval of the merged model
    gate.json             # release-gate verdict; upload only happens if passed
    run_metadata.json     # seed, dataset source, base model, package versions, git SHA
  evaluation/
    base/{metrics.json,predictions.jsonl}
    finetuned/{metrics.json,predictions.jsonl}
    comparison.json
    qualitative_report.json
    qualitative_report.md
```

`data/raw/hdfs/` holds the untouched downloaded archive + extracted files
(git-ignored, like all raw data larger than the toy sample -- see root
`data/README.md`).

## Output schema

```json
{
  "is_anomaly": true,
  "incident_type": "anomaly",
  "confidence": 1.0,
  "evidence": ["<verbatim log line>", "..."],
  "summary": "The HDFS trace exhibits anomalous behavior."
}
```

Ground-truth `confidence` is always 1.0 (it is derived from a hard label);
the model's own generated `confidence` is not calibrated and should be read
qualitatively, not as a calibrated probability.

## Evidence selection (ground truth)

For anomalous traces, up to 5 evidence lines are selected: lines containing
an anomaly-signal keyword (`error`, `exception`, `fail`, `warn`,
`terminating`, `interrupt`, `corrupt`, case-insensitive) first, padded with
the trace's final lines if fewer than 5 keyword hits exist, all substrings of
the actual trace (`src/data/hdfs/sft_format.py`). Nothing is invented.
Evaluation checks this deterministically for model output too
(`evidence_grounding_rate` = fraction of predicted evidence lines that are a
verbatim substring of the input trace, `src/evaluation/hdfs_schema.py`).

## Context length

`max_seq_length` starts at 2048 (`SEQUENCE_LEN` in `.env.example`/the config).
`dataset prepare` reports, via the Qwen tokenizer, what fraction of traces
exceed this in `dataset_stats.json`'s `tokens_over_max_seq_length`. Nothing is
silently truncated by the dataset builder; Axolotl's own `sequence_len`
truncation only applies at train time.

**Measured on the real, full dataset** (575,061 traces, `artifacts/hdfs/processed/dataset_stats.json`):

| | lines | tokens (user content) |
|---|---|---|
| median | 19 | 1,199 |
| p90 | 25 | 1,611 |
| p95 | 28 | 1,827 |
| p99 | 33 | 2,114 |
| max | 298 | 20,953 |

2,048 tokens covers the median trace comfortably (~1,200) and the p95
(1,827), but **1.51% of traces (8,698 blocks) exceed 2,048 tokens**, driven by
a long tail up to an extreme 20,953-token outlier (almost certainly a block
with an unusually large number of log lines, not representative of a typical
trace). Decision for this experiment: keep `SEQUENCE_LEN=2048` and accept
that Axolotl truncates the ~1.5% of traces longer than that at train time --
raising the limit enough to cover the true max would mean a ~10x larger
sequence length, which is not workable on an 8 GB GPU, and the p99 (2,114) is
already barely over 2048, so the practical loss from truncation at 2048 is
small and concentrated in a handful of extreme outliers. If anomaly recall
turns out to correlate with truncated (long) traces in evaluation, that's the
signal to revisit this rather than raising `SEQUENCE_LEN` preemptively.

## Measured training cost (RTX 4060 Laptop, 8 GB)

Measured on this hardware with this config: **~6.5 s per optimizer step**
(micro batch 1 x grad accum 8, so 8 traces per step) and **~4 min per
quick-eval** (1,000 rows at eval batch 1). Effective batch is 8, so
steps = examples / 8 x epochs:

| run | steps | train | + eval | total |
|---|---|---|---|---|
| smoke (`--max-train-samples 200`, 1 epoch) | 25 | ~3 min | ~4 min | **~7 min** |
| 5k subset, 2 epochs | 1,250 | 2.4 h | 0.2 h | **~2.6 h** |
| 20k subset, 2 epochs | 5,000 | 9.7 h | 0.7 h | **~10 h** |
| full 460k, 2 epochs | 115,012 | 222 h | 16 h | **~10 days** |
| full 460k, 3 epochs | 172,518 | 333 h | 24 h | **~15 days** |

So the staged progression isn't optional advice on this hardware -- the full
dataset is a ~10-15 day single-GPU run and belongs on RunPod / a dedicated GPU
(see `runpod/`), while the 5k-20k subset runs are the ones that actually fit an
overnight iteration loop. Use `--max-train-samples` to pick the scale.

**Historical note on why eval frequency matters so much here**: with the
generic template's `eval_steps: 25` pointed at the *full* 57,506-row validation
split, one eval pass took 4.07 hours and would have run ~6,900 times -- roughly
3 years of pure evaluation on a 14-day training job. That is what the
`validation_quick.jsonl` subsample plus `eval_steps: 500` exists to prevent.

## Hardware

Tuned for an 8 GB RTX 4060 laptop GPU / 16 GB RAM / 24 CPU cores:

- 4-bit NF4 QLoRA (`load_in_4bit`, `bnb_4bit_quant_type=nf4`,
  `bnb_4bit_use_double_quant=true`, compute dtype fp16)
- LoRA r=16, alpha=32, dropout=0.05 on all attention + MLP projections
  (`q/k/v/o_proj`, `gate/up/down_proj`)
- micro batch size 1, gradient accumulation 8, gradient checkpointing,
  `paged_adamw_8bit`, SDPA attention, no sample packing
- `bf16: auto` in the Axolotl config -- Axolotl decides bf16 vs fp16 based on
  what the installed CUDA/GPU stack actually supports; this was not hardcoded
  to bf16 since that isn't safe to assume across environments
- `python -m src.cli preflight` reports GPU/VRAM/CUDA/package versions and
  fails before training if VRAM looks insufficient (<6 GiB) or bitsandbytes
  is missing, rather than surfacing a mid-training CUDA OOM

OOM debugging order (matches `docs/OPERATIONS.md`): reduce `SEQUENCE_LEN` first,
then micro batch size (already 1), then increase gradient accumulation or
move to a bigger GPU (RunPod, see `runpod/`). Do not silently drop evaluation
or checkpointing steps to work around OOM.

## Known limitations / caveats

- **Two-class label only.** `normal` / `anomaly` is authoritative; there is
  no finer-grained failure taxonomy in HDFS_v1.
- **Confidence is not calibrated.** It is always 1.0 in training targets;
  don't threshold on the model's predicted confidence as if it were a
  calibrated probability without separately validating it.
- **Config schema is verified, a completed training run is not.** The rendered
  config validates cleanly (no errors, no warnings) against the real
  `AxolotlInputConfig` schema inside the pinned `axolotlai/axolotl:0.18.0`
  image -- verify it yourself after editing the template with:
  ```bash
  docker run --rm -v "$PWD/artifacts/hdfs/runs/<run-id>/train.yml:/tmp/cfg.yml" \
    --entrypoint python fine-tuning-pipeline:0.1.0 -c \
    "import yaml; from axolotl.utils.schemas.config import AxolotlInputConfig; \
     AxolotlInputConfig(**yaml.safe_load(open('/tmp/cfg.yml'))); print('VALID')"
  ```
  What has *not* been demonstrated end to end is a training run that completes,
  merges, and produces evaluated fine-tuned metrics -- run the smoke test first.
- **Baseline model behavior is measured; fine-tuned is not.** A real 4-bit run of
  the base model on this hardware produced 0% schema-valid output on a small
  sample: it classified correctly but emitted `confidence: 95` (out of the [0,1]
  range) and `evidence` as nested objects rather than strings. That is the gap
  fine-tuning is expected to close, and the reason `json_valid` and
  `schema_valid` are tracked as separate metrics rather than one flag.
- **Class imbalance is severe: 97.07% normal / 2.93% anomaly** (16,838 of
  575,061 traces, measured on the real dataset). No oversampling/reweighting
  is applied -- fine-tuning is on the raw distribution. If anomaly recall
  comes out weak, the loss/sampling strategy (not the evaluation gate) is the
  first thing to revisit -- ask before changing the release evaluation gate
  itself.
