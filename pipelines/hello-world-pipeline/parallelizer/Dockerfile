FROM python:3.9-slim

WORKDIR /workspace
COPY pipelines/hello-world-pipeline/parallelizer/* ./
COPY configs/logging.yaml ./configs/logging.yaml
COPY utils ./utils

RUN pip install -r requirements.txt --no-cache-dir