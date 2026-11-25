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

## Intel Arc benchmark

The 2026-11-25 run used Models Configurator revision
`4c38a401b912071e43d2ab4ba2b5f5cebebc1b7e` on an Intel Lunar Lake integrated
GPU (PCI ID `8086:64a0`, `xe` driver, Linux `7.0.0-44-generic`). OpenVINO
2026.4.1 reported `GPU.0` and profile
`8355863b970396ca34421df3f082485ced9d2f842d92e8dd230d2df7c2bc1d53`.
The Vulkan route in Ollama 0.40.16 loaded the F16 model `bge-m3:567m` with
digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`
and reported GPU memory use.

| Measurement | OpenVINO | Ollama |
| --- | ---: | ---: |
| Container start to first dense result, median (range) | 3.11 s (3.03–3.21 s) | 2.50 s (2.41–2.65 s) |
| Warm dense, 1 text, p50 / p95 | 12.50 / 14.35 ms | 43.40 / 45.39 ms |
| Warm dense, 8 texts, p50 / p95 | 102.23 / 140.72 ms | 330.14 / 343.29 ms |
| Warm dense throughput, 8 texts per request | 73.12 texts/s | 22.92 texts/s |
| Warm Dense + Sparse + ColBERT, 1 text, p50 / p95 | 18.42 / 23.44 ms | Unsupported |
| Peak container anonymous memory | 617 MiB | 296 MiB |
| Peak container working set | 992 MiB | 361 MiB |
| Peak allocation attributed to Xe DRM clients | 1,133 MiB | 1,265 MiB |

Each cold measurement started a fresh container without clearing the host file
cache. The warm workloads used one warm-up and 20 sequential requests to the
worker's Unix socket or Ollama's loopback HTTP endpoint. Batch 1 used the first
text below and batch 8 used all eight in order. The rich request used the same
single text. Memory values are peaks across the three runs sampled every 100 ms.
The working set excludes inactive file cache. The integrated GPU shares system
memory. The container and DRM figures overlap, and DRM allocations are distinct
from Ollama's own memory estimate.

1. Администратор может изменить лимит платежа.
2. Оператор не может изменить лимит платежа.
3. Система сохраняет журнал операций в течение 30 дней.
4. Система удаляет журнал операций через 90 дней.
5. Заказ можно оплатить банковской картой.
6. Возврат средств доступен после отмены заказа.
7. Сервис возвращает ошибку E104 при повторной отправке.
8. Уведомление отправляется только после подтверждения.

A separate check through the running manager compared three short Russian texts
against FlagEmbedding 1.2.11 on CPU FP32 with the original BAAI revision
`5617a9f61b028005a4858fdac845db406aefb181`. The minimum dense cosine similarity
was 0.9999978. The sparse token IDs matched and the largest absolute weight
difference was 0.0005451. The minimum per-token ColBERT cosine was 0.9999882.
The fixed texts were the first text above, the same sentence followed by
“дважды” before the final period, and “Вернуть ошибку E104 при повторной отправке.”

For a long-running service on this Arc GPU, OpenVINO is the faster tested route
for dense embeddings and also provides both projection heads. Ollama reached its
first result sooner and used less anonymous container memory. The benchmark
covers short texts and a single machine. Retrieval quality and the cost of
ranking candidates with the extra representations are tracked in Project
Creator. [Runtime evaluation](https://github.com/VolkovDani/models-configurator/issues/9) · [Retrieval evaluation](https://github.com/VolkovDani/project-creator/issues/64)
