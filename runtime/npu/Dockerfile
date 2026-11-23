FROM ubuntu:26.04@sha256:d66b366dac61b575bee2aef07e9c26e8070096fbe84a3d4d3eb0da069b5d76db
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libtbb12 python3-venv && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/models/runtime/npu
COPY requirements.txt requirements-installer.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip --disable-pip-version-check install --no-deps --no-cache-dir -r requirements-installer.txt && \
    /opt/venv/bin/pip --disable-pip-version-check install --no-deps --only-binary=:all: --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip check
COPY npu_runtime.py .
ENV PYTHONDONTWRITEBYTECODE=1 HOME=/tmp
USER 65532:65532
ENTRYPOINT ["/opt/venv/bin/python", "/opt/models/runtime/npu/npu_runtime.py"]
