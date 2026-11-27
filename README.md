# Models Configurator

[Русская версия](README.ru.md)

A same-host model service for applications that need structured chat or embeddings. The Go server includes a status panel, a settings editor, a versioned API, adapters for OpenVINO NPU, Ollama and OpenAI-compatible providers. Each request names one configured profile. The server has no provider fallback.

## Start with an empty inventory

Use Docker Compose 2.24+ on Linux. On the development branch:

```bash
test -e models.json || cp models.example.json models.json
docker compose up --build -d
curl http://127.0.0.1:8317/v1/models
```

Open http://127.0.0.1:8317/ for the status page. This configuration starts only the manager. To run from source, use Go 1.26+ and `go run ./cmd/models-configurator -config models.json`. The native server binds to loopback by default.

## Add models

For local CPU inference, explicitly start Ollama and pull the weights you choose.

```bash
docker compose --profile cpu up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull bge-m3:567m
```

Read the full digests as reported by Ollama's private `/api/tags` endpoint. The example uses `jq` to display tags and digests and keeps the Ollama port private.

```bash
ollama_container=$(docker compose ps -q ollama)
docker run --rm --network "container:$ollama_container" \
  alpine:3.22@sha256:49340b346798875b5592e5d420241f2f9cfba5f47e62ebb1446a1b34734e898f \
  wget -qO- http://127.0.0.1:11434/api/tags |
  jq -r '.models[] | [.name, .digest] | @tsv'
```

Edit the private `models.json`. Specify the Ollama tag explicitly, including `:latest` when applicable. Replace the digest placeholders with those full values. Keep only the entries you intend to use. The external endpoint below is an example; select the deployment and its version explicitly.

```json
{
  "models": [
    {"name":"local-chat","provider":"ollama","base_url":"http://ollama:11434","model":"qwen2.5:7b","role":"chat","digest":"<full-qwen-sha256>"},
    {"name":"local-embed","provider":"ollama","base_url":"http://ollama:11434","model":"bge-m3:567m","role":"embeddings","digest":"<full-bge-sha256>"},
    {"name":"hosted-chat","provider":"openai-compatible","base_url":"https://provider.example/v1","model":"deployment-v1","revision":"snapshot-1","role":"chat","key_env":"HOSTED_API_KEY"}
  ]
}
```

For an authenticated upstream, set the named variable in a private `providers.env` (Compose) or in the server process environment (native). Use HTTPS when sending credentials or content across untrusted networks. Recreate the manager after configuration or secret changes: `docker compose up -d --force-recreate models-configurator`. Replacing weights behind an external deployment requires the operator to change its revision. Ollama digests are checked before and after inference. Persistent weights are stored in the manager's Ollama volume.

For GPU inference, see the [Ollama GPU profiles and migration guide](docs/ollama-gpu.md).

## Web panel appearance

The model inventory, settings, and call history follow your system theme by default. Use **Theme** in the page header to choose **System**, **Light**, or **Dark**. Your browser remembers the choice across those pages, including the history chart.

## Settings in the web panel

Open **Model settings** from the model inventory. Set the manager concurrency limit, rename profiles and choose which should be available to clients. For Ollama you can choose an installed local model with a verified digest and select chat or embeddings if the model supports that role. Existing decision profiles keep their assigned model and role. Chat profiles expose the context, response token limit, temperature and top P. Explicit request options take precedence over the Ollama chat defaults.

Choose **Save settings file** and put the downloaded `model-settings.json` next to `compose.yml`. Add `compose.settings.yml` to your Compose files and recreate `models-configurator`. For a native launch, set `MODELS_SETTINGS_PATH` to the saved file. The settings file takes precedence over `MODELS_MAX_PARALLEL_REQUESTS`. Models Configurator validates settings against the deployment's current inventory at startup. After changing this inventory, save a matching settings file or start without the settings overlay. To restore the original settings in Compose, omit `compose.settings.yml` and recreate the manager.

The base `models.json` still defines providers, endpoints and credential references. The editor exports a separate file that applies at restart. Check `/v1/models` afterward and update clients when model IDs change. [Change NPU limits](docs/npu.md#change-npu-token-limits-in-the-web-panel).

## Connect clients

Open [Swagger UI](http://127.0.0.1:8317/docs/) from the status panel or directly to browse endpoint schemas, review examples and try requests. Download [OpenAPI JSON](http://127.0.0.1:8317/openapi.json) to import the contract into a client tool. The service includes the UI assets and works offline.

Read an immutable profile ID from `GET /v1/models` and send it in the request's `model` field. Local clients use `http://127.0.0.1:8317/v1`. Containers can join the `models-configurator-clients` network and use `http://models-configurator:8317/v1`. Only attach trusted applications: clients on this network can use all configured profiles, including billable external ones. For Project Creator select the openai-compatible runtime for each active role, paste its profile ID, and set `PROJECT_CREATOR_MODELS_NETWORK=models-configurator-clients` in its private configuration.

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | List configured profiles without contacting upstreams. |
| `POST /v1/chat/completions` | Non-streaming messages and optional strict JSON Schema output. |
| `POST /v1/chat/token-count` | Count or estimate chat input tokens and return the applicable input and completion budgets before generation. |
| `POST /v1/embeddings` | Float vectors for input text, with the selected profile's dimension. Project Creator needs 1024 components. |
| `GET /status`, `GET /` | Inspect provider metadata, availability and last check. The status page never runs inference. Ollama installation and current loading are reported separately. |

For example, run a second local client with a configured embedding ID:

```bash
curl -sS http://127.0.0.1:8317/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"<embedding-profile-id>","input":["hello"],"encoding_format":"float"}'
```

[Run local decision models](docs/decision-models.md) with System One D3 or Laya.

### Check the chat token budget

Send the intended completion body to `POST /v1/chat/token-count`, including the full profile ID, messages and any output schema or completion cap. The response contains `model`, `input_tokens`, `max_input_tokens`, `max_completion_tokens` and `exact`. A successful response can report input above the budget. Clients can reduce the context and count again before generating.

- OpenVINO NPU uses the pinned chat template and tokenizer locally and returns `exact: true`. Supply a strict JSON Schema and omit sampling overrides. The input limit comes from the manifest and the output limit is capped by the manifest. [NPU limits and diagnostics](docs/npu.md#api-and-diagnostics)
- Ollama with an explicit `ollama_chat.context_tokens` returns `exact: false` using message and optional schema bytes with allowances for chat formatting. The configured context minus the requested or default completion limit is the input budget. The estimate is conservative for typical templates and actual model limits still apply during generation.
- Ollama without an explicit context or sufficient room for the output reserve and OpenAI-compatible profiles return HTTP 501 with `TOKEN_COUNT_UNSUPPORTED`. Use a client-side fallback when counting is unavailable. Token counting runs without generating a completion or contacting external providers.

## Model call history

Open **Model call history** from the status panel or visit http://127.0.0.1:8317/history. The table shows the start time (UTC), pinned model profile and provider, operation, result, HTTP status, duration, and reported input and output token counts. Duration starts when the adapter receives the request, after any queue wait. A dash marks an unavailable count. Scroll down or use **Load more** for earlier calls.

The **Duration and tokens** chart plots token count on the horizontal axis and call duration in milliseconds on the vertical axis. Select an operation and an available count (input, output, or total) to compare profiles. The total requires both counts. Each point represents one call with the selected count. The chart updates as earlier calls load and runs locally from an embedded Recharts bundle.

The manager records chat, embedding, decision, and BGE-M3 calls after they reach the adapter. Refusals and failures appear in the table. Token budget checks, provider probes, invalid requests, and requests rejected or canceled while waiting are excluded. When available, the manager records counts reported by Ollama, the compatible provider, or the local runtime. Embedding and BGE-M3 calls have an input count when available. Older records and workers without counts show a dash. History stores only operational metadata and token counts. Trusted clients with network access can read it.

For programmatic access, use `GET /history.json`. Each record may include `input_tokens` and `output_tokens`. The response returns up to 50 calls in descending record order in `calls`. If it contains `next_before`, pass it as `?before=<next_before>` to get older calls.

The most recent 10,000 records are available across restarts. Compose stores them in the `model_history` volume, which survives container replacement. On the host, the default is `logs/model-history.jsonl` relative to the working directory. Set `MODELS_HISTORY_PATH` to use a different file. Each manager process needs its own file. After startup, write failures are logged and inference responses remain available.

### Clear model call history

`models-configurator history clear` moves the entire history file into a private backup directory next to the original and prints the backup path. The manager starts with empty history on its next launch. If history is already empty, the command succeeds without creating a backup. The command uses a file lock shared with the manager to prevent clearing while the manager is writing history. Native CLI clearing is available on Linux, macOS, DragonFly BSD, FreeBSD, NetBSD, and OpenBSD.

For Docker Compose, first rebuild and recreate the manager if you are upgrading from a version without this command. Use the same Compose project, files, and environment as the running manager.

```bash
docker compose stop models-configurator &&
  docker compose run --rm --no-deps models-configurator history clear &&
  docker compose start models-configurator
```

For a native launch, stop the manager and run `models-configurator history clear` from its working directory with the same `MODELS_HISTORY_PATH`, if set. When running from source, use `go run ./cmd/models-configurator history clear`. Then start the manager again.

## Concurrent requests

The manager runs one inference or token count request at a time by default. To allow two independent runtimes to work at once, set the following in the manager's private `.env` and apply it with `docker compose up -d models-configurator`. For a native launch, export the variable before starting the server.

```dotenv
MODELS_MAX_PARALLEL_REQUESTS=2
```

Choose a positive integer that your hardware and providers can sustain. Each OpenVINO runtime socket serves one request at a time even if several profiles point to it. A request for a free runtime can proceed while another waits for its socket. Up to 128 requests may wait in each manager process. A full queue returns HTTP 429. Waiting counts toward the two-minute request deadline, and disconnecting while queued removes the request.

## Intel NPU

Run the OpenVINO worker in Docker or on the host and select its pinned generation profile from any trusted manager client. An existing Project Creator export can be mounted read-only. A fresh installation can fetch the pinned Qwen snapshot from Hugging Face and export it offline. [NPU setup](docs/npu.md) · [Prepare a model](docs/npu-model.md)

[Run BGE-M3 on Intel Arc](docs/bge-arc.md) for dense embeddings and optional sparse and ColBERT representations.

Run `go test -race ./...`, `go vet ./...`, `python3 -m unittest discover -s runtime/npu` and `python3 -m unittest discover -s tests` for deterministic checks.

### Rebuild the history chart

The checked-in bundle lets Go and Docker builds serve the chart offline. To update it after changing the chart source, use Node.js 26 and Bun 1.4.4.

```bash
cd frontend
bun install --frozen-lockfile
bun run format:check
bun run typecheck
bun run test
bun run build
```

Commit the regenerated bundle and third-party notices in `internal/status/chart/`. CI verifies that both match the sources and lockfile.
