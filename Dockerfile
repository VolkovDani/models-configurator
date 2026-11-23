FROM golang:1.27.4-bookworm AS builder
WORKDIR /src
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/models-configurator ./cmd/models-configurator

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /out/models-configurator /models-configurator
USER 65532:65532
ENTRYPOINT ["/models-configurator"]
