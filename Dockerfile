FROM golang:1.27.4-bookworm AS builder
WORKDIR /src
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/models-configurator ./cmd/models-configurator \
    && install -d -m 0700 /out/state

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /out/models-configurator /models-configurator
COPY --from=builder --chown=65532:65532 /out/state/ /var/lib/models-configurator/
ENV MODELS_HISTORY_PATH=/var/lib/models-configurator/history.jsonl
USER 65532:65532
ENTRYPOINT ["/models-configurator"]
