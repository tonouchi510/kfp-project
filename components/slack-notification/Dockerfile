FROM debian:buster-slim

WORKDIR /workspace
COPY utils/slack-notification/entrypoint.sh ./entrypoint.sh

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	curl && \
    cp entrypoint.sh /usr/local/bin && \
    chmod 755 /usr/local/bin/entrypoint.sh && \
	apt-get autoremove && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]