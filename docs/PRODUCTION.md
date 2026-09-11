# Production checklist

## Repository

- Protect `main`.
- Require CI to pass.
- Require review for training configuration changes.
- Keep secrets only in GitHub/secret storage.
- Keep raw proprietary datasets outside Git if large or sensitive.

## Runner

Use Linux with Docker + NVIDIA Container Toolkit. Give the runner custom label `gpu` and route only GPU jobs to it.

Do not use an everyday developer laptop as the long-lived production runner.

## Containers

The Dockerfile pins an Axolotl release tag (`0.18.0`) rather than floating `main-latest`. After validating the image in your organization, pin the exact image digest for stricter supply-chain reproducibility.

## Model registry

Use a private Hugging Face repository for proprietary models. Record the training Git SHA, dataset fingerprint, and evaluation report with each release.

## Deployment

For a service, load the merged model in a dedicated serving image (vLLM/TGI/SGLang depending on the model) rather than running the training image in production.

## Rollback

Every production model must have an immutable version identifier and a previous known-good version. Roll back by selecting the previous model artifact, not by rebuilding from memory.
