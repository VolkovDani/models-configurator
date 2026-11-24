FROM openvino/ubuntu24_dev@sha256:e8b0d380b45ad923c6c493eddcf8fca374b8a07708f895b3f558e0b2ba1ca933
# Keep the image's Intel GPU driver, use only the pinned Python inference stack.
USER root
WORKDIR /opt/models
ENV PYTHONPATH="" LD_LIBRARY_PATH="" PYTHONDONTWRITEBYTECODE=1 HOME=/tmp HF_HUB_OFFLINE=1 DO_NOT_TRACK=1
COPY runtime/npu/requirements-installer.txt runtime/bge/requirements.txt ./
RUN /usr/bin/python3 -m venv /opt/bge && \
    /opt/bge/bin/pip install --disable-pip-version-check --no-cache-dir --no-deps -r requirements-installer.txt && \
    /opt/bge/bin/pip install --disable-pip-version-check --no-cache-dir --no-deps --only-binary=:all: -r requirements.txt && \
    /opt/bge/bin/pip check
COPY runtime/npu/npu_runtime.py runtime/npu/
COPY runtime/bge/server.py runtime/bge/engine.py runtime/bge/
USER 65532:65532
ENTRYPOINT ["/opt/bge/bin/python", "-m", "runtime.bge.server"]
