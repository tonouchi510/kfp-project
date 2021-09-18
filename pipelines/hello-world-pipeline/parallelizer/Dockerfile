FROM python:3.9-slim

WORKDIR /workspace
COPY pipelines/hello-world-pipeline/parallelizer/* ./
RUN pip install -r requirements.txt --no-cache-dir