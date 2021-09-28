FROM python:3.9-slim

WORKDIR /workspace
COPY pipelines/hello-world-pipeline/hello/* ./
COPY configs/logging.yaml ./configs/logging.yaml
COPY utils ./utils

RUN pip3 install -r requirements.txt
