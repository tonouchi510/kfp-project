FROM python:3.9-slim
LABEL Name=gcp-client maintainer=tonouchi27@gmail.com

SHELL ["/bin/bash", "-c"]
ENV PATH $PATH:/tools/node/bin:/tools/google-cloud-sdk/bin

WORKDIR /workspace
COPY components/slack-notification/entrypoint.sh ./entrypoint.sh

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	git \
	wget \
    curl \
	unzip \
	libsm6 \
    libgl1-mesa-dev \
	pigz \
	libxext6 \
	libglib2.0-0 \
	libxrender1 && \
    cp entrypoint.sh /usr/local/bin && \
    chmod 755 /usr/local/bin/entrypoint.sh && \
	wget -nv https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.zip && \
    unzip -qq google-cloud-sdk.zip -d /tools && \
    rm google-cloud-sdk.zip && \
    /tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    /tools/google-cloud-sdk/bin/gcloud -q components update \
        gcloud core gsutil && \
    /tools/google-cloud-sdk/bin/gcloud config set component_manager/disable_update_check true && \
    touch /tools/google-cloud-sdk/lib/third_party/google.py && \
	apt-get autoremove && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
