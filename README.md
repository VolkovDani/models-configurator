# Models Configurator

[Русская версия](README.ru.md)

A same-host model service for applications that need structured chat or embeddings. The Go server includes a read-only status page, a versioned API, an Ollama adapter and an OpenAI-compatible adapter. Each request names one configured profile. The server has no provider fallback.

## Start with an empty inventory

Use Docker Compose 2.24+ on Linux. On the development branch:

```bash
cp -n models.example.json models.json
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

The pilot covers same-host deployment, CPU Ollama and explicit external APIs. GPU/NPU installation and specialized classifiers remain in Project Creator during migration. Run `go test -race ./...`, `go vet ./...` and `python3 tests/test_compose.py` for deterministic checks.
