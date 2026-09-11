ARG AXOLOTL_IMAGE=axolotlai/axolotl:0.18.0
FROM ${AXOLOTL_IMAGE}

WORKDIR /workspace/fine-tuning-pipeline

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface/hub

COPY requirements.txt /workspace/fine-tuning-pipeline/requirements.txt

RUN python -m ensurepip --upgrade && \
    python -m pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/base-image-constraints.txt && \
    python -m pip install --no-cache-dir -c /tmp/base-image-constraints.txt -r requirements.txt

COPY . /workspace/fine-tuning-pipeline
RUN chmod +x scripts/*.sh

ENTRYPOINT ["/workspace/fine-tuning-pipeline/scripts/run_pipeline.sh"]