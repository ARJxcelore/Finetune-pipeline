# Day-to-day operation

## Normal development

```bash
make validate
make test
make build
make train
```

## Training a new dataset

Replace `data/raw/dataset.jsonl`, validate it, and commit the dataset pointer/metadata rather than secrets.

For large production datasets, modify `src/data/prepare.py` to read your object store or data platform instead of putting the full data in Git.

## Launching a production run

Use GitHub Actions -> Fine-tune Model -> Run workflow. Select the model and hyperparameters. The workflow validates first and trains only on a labeled GPU runner.

## Debugging OOM

First reduce sequence length. Then reduce micro batch size (already 1), increase gradient accumulation, disable optional optimizations, or move the run to a larger GPU. Do not solve an OOM by silently deleting evaluation or checkpointing.

## Releasing

Only upload models after the evaluation gate passes. For a strict release process, add an approval step before changing the production model pointer.
