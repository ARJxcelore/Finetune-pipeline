# RunPod setup

## Option A: Pod as a GPU runner

Create a Linux GPU Pod with Docker and NVIDIA support, clone this repository, install the GitHub Actions self-hosted runner, and give it the custom label `gpu`.

Then the existing `.github/workflows/train.yml` can route training to that runner.

## Option B: RunPod GitHub integration

RunPod can import a repository and build/deploy its Docker image. This is more suitable for inference workers than long-running training, but the same versioned image workflow can be used for specialized jobs.

For training, a dedicated Pod is usually simpler because the pipeline is a long-running batch workload.

## Operational rules

- Prefer versioned images, not `latest`.
- Prefer persistent/network storage for model caches when the provider configuration supports it.
- Terminate GPU resources after batch training.
- Keep credentials in RunPod/GitHub secrets.
