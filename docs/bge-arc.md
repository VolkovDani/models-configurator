# BGE-M3 on Intel Arc

[Русская версия](bge-arc.ru.md)

This opt-in profile runs the pinned BAAI BGE-M3 FP16 encoder and both trained projection heads on an Intel GPU through OpenVINO. The worker has no network, mounts the model read-only, and exposes a private Unix socket. Tokenization and sparse token aggregation run on the CPU.

## Prepare the profile

Use the pinned FP16 encoder export from the [Project Creator device verification](https://github.com/VolkovDani/project-creator/blob/94003dd680fbf72812f4145d1345fbb34bfa44b1/docs/intel-device-verification.md). Its OpenVINO IR and tokenizer must match the SHA-256 values in the preparation tool. Run the tool with a local Python environment containing NumPy and PyTorch. The Project Creator export environment used for the verification contains both.

```sh
pc_dir=/absolute/path/to/project-creator
test -e models.json || cp models.example.json models.json
export MODELS_BGE_MODEL_DIR="$PWD/.bge/models/bge-m3-arc"
export MODELS_BGE_PROFILE=$(
  "$pc_dir/.npu/export-env/bin/python" -m runtime.bge.prepare \
    --encoder-dir "$pc_dir/.npu/models/bge-m3-fp16" \
    --output-dir "$MODELS_BGE_MODEL_DIR" --download-heads
)
```

The command fetches only the two projection heads from the pinned BAAI revision. Use `--heads-dir /absolute/path/to/heads` with the official `sparse_linear.pt` and `colbert_linear.pt` for offline preparation. The output directory must be new. The printed SHA-256 pins the weights, tokenizer, preprocessing and token limits.

## Start the runtime

Add an entry to the private `models.json`. Replace the digest with the printed value.

```json
{"name":"bge-arc","provider":"openvino-bge","base_url":"unix:///run/models-bge/runtime.sock","model":"BAAI/bge-m3","role":"embeddings","digest":"<profile SHA-256>"}
```

The supplied overlay expects the Intel Arc device at `/dev/dri/renderD128`. Check the render node on the host. Persist the profile settings in the local `.env` if the deployment should survive a shell restart.

```sh
export MODELS_BGE_SOCKET_DIR="$PWD/.bge/run"
install -d -m 0750 "$MODELS_BGE_SOCKET_DIR"
export MODELS_BGE_UID=$(id -u)
export MODELS_BGE_SOCKET_GID=$(stat -c %g "$MODELS_BGE_SOCKET_DIR")
export MODELS_BGE_DEVICE_GID=$(stat -c %g /dev/dri/renderD128)
docker compose -f compose.yml -f compose.bge-arc.yml up --build -d
curl --fail --silent http://127.0.0.1:8317/v1/models
```

For an installation already using the NPU overlay, append `compose.bge-arc.yml` to its existing Compose files. If an unclean stop leaves the socket behind, verify that its owning worker is stopped before removing the stale socket.

## Use the profile

Use the full model ID from the inventory. The existing `POST /v1/embeddings` endpoint returns 1024-component dense vectors and accepts up to 32 texts of at most 4096 tokens each. For experiments, send the following body to `POST /v1/bge-m3/encode`. That endpoint accepts up to four texts and requires an explicit list of modes. Sparse or ColBERT requests are limited to 128 tokens per text. All token caps include special tokens.

```json
{"model":"<manager model ID>","input":["Отправить заявку."],"modes":["dense","sparse","colbert"]}
```

The response carries the manager model ID and one data row per input. Sparse output maps vocabulary IDs to positive weights. ColBERT output contains vectors per token. Changing Project Creator's embedding profile schedules reindexing. Its current search uses dense vectors.
