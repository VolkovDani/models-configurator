# Models Configurator

[Русская версия](README.ru.md)

A same-host model service for applications that need structured chat or embeddings. The Go server includes a read-only status page, a versioned API, adapters for OpenVINO NPU, Ollama and OpenAI-compatible providers. Each request names one configured profile. The server has no provider fallback.

## Start with an empty inventory

Use Docker Compose 2.24+ on Linux. On the development branch:

```bash
test -e models.json || cp models.example.json models.json
docker compose up --build -d
curl http://127.0.0.1:8317/v1/models
```

Open http://127.0.0.1:8317/ for the status page. This configuration starts only the manager. To run from source, use Go 1.26+ and `go run ./cmd/models-configurator -config models.json`. The native server binds to loopback by default.

## Add models

For local CPU inference, explicitly start Ollama and pull the weights you choose. The following commands also print the full SHA-256 digests of the downloaded manifests:

```bash
docker compose --profile cpu up -d ollama
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull bge-m3:567m
docker compose exec ollama sha256sum /root/.ollama/models/manifests/registry.ollama.ai/library/{qwen2.5/7b,bge-m3/567m}
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

## Connect clients

Read an immutable profile ID from `GET /v1/models` and send it in the request's `model` field. Local clients use `http://127.0.0.1:8317/v1`. Containers can join the `models-configurator-clients` network and use `http://models-configurator:8317/v1`. Only attach trusted applications: clients on this network can use all configured profiles, including billable external ones. For Project Creator select the openai-compatible runtime for each active role, paste its profile ID, and set `PROJECT_CREATOR_MODELS_NETWORK=models-configurator-clients` in its private configuration.

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/models` | List configured profiles without contacting upstreams. |
| `POST /v1/chat/completions` | Non-streaming messages and optional strict JSON Schema output. |
| `POST /v1/embeddings` | Float vectors for input text, with the selected profile's dimension. Project Creator needs 1024 components. |
| `GET /status`, `GET /` | Inspect provider metadata, availability and last check. The status page never runs inference. Ollama installation and current loading are reported separately. |

For example, run a second local client with a configured embedding ID:

```bash
curl -sS http://127.0.0.1:8317/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"<embedding-profile-id>","input":["hello"],"encoding_format":"float"}'
```

[Run local decision models](docs/decision-models.md) with System One D3 or Laya.

## Model call history

Open **Model call history** from the status panel or visit http://127.0.0.1:8317/history. The page shows when each call started (UTC), the pinned model profile and provider, the operation, its outcome and HTTP status, and the time spent executing the call, excluding queue wait. Scroll down or use **Load more** for earlier calls.

The manager records chat, embedding, decision, and BGE-M3 requests once they leave the queue and reach the adapter. Refusals and failures appear in history. Provider probes, invalid requests, and requests rejected or canceled while waiting do not appear. The history contains operational metadata only. It stores no prompts, outputs, third-party error text, credentials, caller identifiers, or token counts. Like the rest of the local service, the history is shared with applications that have trusted network access to the manager.

For programmatic access, use `GET /history.json`. It returns up to 50 calls in descending record order in `calls`. If the response contains `next_before`, pass it as `?before=<next_before>` to get older calls.

The most recent 10,000 records are available across restarts. Compose stores them in the `model_history` volume, which survives container replacement. On the host, the default is `logs/model-history.jsonl` relative to the working directory. Set `MODELS_HISTORY_PATH` to use a different file. Each manager process needs its own file. After startup, write failures are logged and inference responses remain available.

## Concurrent requests

The manager runs one inference request at a time by default. Chat, embedding and decision requests share the limit. To allow two independent runtimes to work at once, set the following in the manager's private `.env` and apply it with `docker compose up -d models-configurator`. For a native launch, export the variable before starting the server.

```dotenv
MODELS_MAX_PARALLEL_REQUESTS=2
```

Choose a positive integer that your hardware and providers can sustain. Each OpenVINO runtime socket serves one generation at a time even if several profiles point to it. A request for a free runtime can proceed while another waits for its socket. Up to 128 requests may wait in each manager process. A full queue returns HTTP 429. Waiting counts toward the two-minute request deadline, and disconnecting while queued removes the request.

## Intel NPU

Run the OpenVINO worker in Docker or on the host and select its pinned generation profile from any trusted manager client. An existing Project Creator export can be mounted read-only. A fresh installation can fetch the pinned Qwen snapshot from Hugging Face and export it offline. [NPU setup](docs/npu.md) · [Prepare a model](docs/npu-model.md)

[Run BGE-M3 on Intel Arc](docs/bge-arc.md) for dense embeddings and optional sparse and ColBERT representations.

Run `go test -race ./...`, `go vet ./...`, `python3 -m unittest discover -s runtime/npu` and `python3 -m unittest discover -s tests` for deterministic checks.
