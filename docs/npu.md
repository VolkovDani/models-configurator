# Intel NPU generation

[Русская версия](npu.ru.md)

Models Configurator runs OpenVINO GenAI directly on Intel AI Boost. The application submits messages and a strict JSON schema to the manager. The manager verifies the selected manifest profile and calls its local runtime over a Unix socket. Project Creator owns comparison prompts, retries, source mapping and validation. The NPU profile serves generation; configure embeddings separately.

## Use an existing Project Creator export

Run these commands in the Models Configurator checkout on the NPU host. This example reuses the Qwen2.5-7B INT4 weights and version 2 manifest from a sibling Project Creator checkout. For a fresh installation, [prepare the pinned model](npu-model.md) and set the model directory, manifest and profile to its output paths.

```bash
pc_dir="$(cd ../project-creator && pwd)"
export MODELS_NPU_MODEL_DIR="$pc_dir/.npu/models/qwen2.5-7b-int4"
export MODELS_NPU_MANIFEST="$pc_dir/.npu/profile.json"
export MODELS_NPU_PROFILE="$(cat "$pc_dir/.npu/profile.sha256")"
export MODELS_NPU_UID="$(id -u)"
export MODELS_NPU_SOCKET_GID="$(id -g)"
export MODELS_NPU_DEVICE_GID="$(stat -c %g /dev/accel/accel0)"
export MODELS_NPU_SOCKET_DIR="$PWD/.npu/run"
export MODELS_NPU_CACHE_DIR="$PWD/.npu/cache"
export MODELS_NPU_LOG_DIR="$PWD/logs/npu"
install -d -m 0750 "$MODELS_NPU_SOCKET_DIR" "$MODELS_NPU_CACHE_DIR" "$MODELS_NPU_LOG_DIR"
printf '%s\n' "$MODELS_NPU_PROFILE"
```

Use an unprivileged Linux account and a socket group trusted to run inference. Add this entry to the `models` array in your private `models.json`, substituting the full profile hash printed above. An empty inventory template is in `models.example.json`. The digest identifies the canonical manifest, including its model file hashes and runtime pins.

```json
{"name":"qwen-npu","provider":"openvino-npu","model":"Qwen/Qwen2.5-7B-Instruct","base_url":"unix:///run/models-npu/runtime.sock","role":"chat","digest":"<64 lowercase hex characters>"}
```

## Start the container runtime

The example targets native Linux, Ubuntu 26.04 amd64 and a working Intel NPU driver. The host supplies the kernel module, firmware, device node and the four userspace libraries listed in `compose.npu.yml`. Only the runtime gets the NPU device. It has no network, mounts the manifest and weights read-only and uses its own writable cache. The manager gets read access to the socket directory. Pin or recheck the driver combination when updating the host.

```bash
export COMPOSE_FILE=compose.yml:compose.npu-host.yml:compose.npu.yml
docker compose up --build -d
docker compose logs --tail=20 npu-runtime
curl -fsS http://127.0.0.1:8317/status
curl -fsS http://127.0.0.1:8317/v1/models
```

Wait for the runtime's ready log entry and the profile's `loaded: true`. Copy the full `openvino-npu.…` ID from the inventory for the client. A cold start includes hashing the weights and compiling the model. Retain these variables in the shell or set them in the manager's private `.env` for subsequent `docker compose up -d` and `docker compose stop` commands.

## Native runtime with a manager container

Use the same configuration and directories. Start the runtime in one terminal. A matching existing Project Creator runtime interpreter can be selected directly. For a new environment use `scripts/install-npu-runtime .npu/venv` and substitute its Python executable below.

```bash
"$pc_dir/.npu/runtime-env/bin/python" runtime/npu/npu_runtime.py serve \
  --model "$MODELS_NPU_MODEL_DIR" --manifest "$MODELS_NPU_MANIFEST" \
  --profile "$MODELS_NPU_PROFILE" --socket "$MODELS_NPU_SOCKET_DIR/runtime.sock" \
  --cache "$MODELS_NPU_CACHE_DIR"
```

In another terminal with the same configuration run `docker compose -f compose.yml -f compose.npu-host.yml up --build -d`. One runtime owns each socket. Each new worker verifies the pinned artifacts, selected device and package versions. The CLI preserves existing model directories. Stop a native runtime with Ctrl+C, or stop the container runtime through Compose.

## API and diagnostics

Use `POST /v1/chat/completions` with the full manager ID, messages and `response_format.type=json_schema` with `strict: true`. The pinned NPU uses greedy decoding. Streaming and sampling overrides are unsupported. The request token cap is an upper bound within the manifest's limit. Oversized prompts and incomplete generations fail explicitly. The private socket accepts up to 65,536 request bytes. Domain-specific checks belong to the caller.

Disconnecting an active client stops its native worker. The next generation request waits for the previous worker to stop before its health check can reload the same verified profile. The status panel reads metadata without loading weights or running inference. The manager returns an opaque `X-NPU-Call-ID` for a managed chat request. Match it to the runtime lifecycle log. Provider failures expose only allowlisted codes. Use the runtime logs and `logs/npu/runtime-lifecycle.jsonl` to diagnose startup, cancellation and recovery. A worker crash, generation timeout or changed artifact requires operator intervention.

Project Creator can use this chat profile with local Arc embeddings. Configure its private environment with the manager's shared network, the `openai-compatible` generator runtime, the URL `http://models-configurator:8317/v1` and the full model ID. Use `./project-creator setup --device arc --switch-device` when switching from a previous local NPU setup. See the [client setup guide](https://github.com/VolkovDani/project-creator/blob/develop/docs/setup-cli.en.md).

## Hardware verification on 2026-11-23

On an Intel Core Ultra 7 268V (Intel AI Boost, Ubuntu 26.04.1, Intel userspace and firmware 1.38.0, Level Zero loader 1.32.0), the offline, non-root Docker worker loaded the existing Qwen INT4 manifest `f36b13cbad8752f05d141eba6f817045986a615d13ffb5688fe417faeebca271`. With identical inputs, the three Project Creator fixtures returned the same full findings as the native legacy worker. The NPU busy counter increased on each generation. On disconnect the worker stopped in 0.58 s; the idle status query reported unloaded, and the next request completed with the expected finding 19.76 s after cancellation. The native baseline recovered and completed in 15.52 s. These are individual observations on this device. The consumer regression suite uses Project Creator revision `e94a4ff1963689c33ea95e57c3d6b466e15ac891`. [Reproduce the client checks](https://github.com/VolkovDani/project-creator/blob/e94a4ff1963689c33ea95e57c3d6b466e15ac891/docs/adr/012-standalone-model-manager.md)
