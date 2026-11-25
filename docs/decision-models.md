# Local decision models

[Русская версия](decision-models.ru.md)

Configure a model with role `decision` to expose `POST /v1/decisions`. The manager supports local Ollama 0.40.16 decision models pinned by the digest reported by `/api/tags`. Project Creator supplies the question, interprets the returned model scores and owns review policy.

## Start a local Ollama runtime

The base Compose file includes an opt-in CPU profile. To use Intel Arc, use Docker Compose 2.24.4 or later, add `compose.ollama-arc.yml` to the files already in your configuration and enable the `arc` profile. The example below is for an installation with the base file only.

```bash
export COMPOSE_FILE=compose.yml:compose.ollama-arc.yml
export COMPOSE_PROFILES=arc
export MODELS_OLLAMA_DRI_DEVICE=/dev/dri/renderD128
export MODELS_OLLAMA_RENDER_GID=$(stat -c %g "$MODELS_OLLAMA_DRI_DEVICE")
docker compose up -d --build ollama
docker compose exec ollama ollama pull d3:4b
```

Use the render node that belongs to your Intel GPU. Save the settings in the manager's private `.env` when using Project Creator's linked CLI. For CPU, use the base file and the `cpu` profile. The Ollama service uses a private runtime network and its own persistent model volume. For an existing pinned installation, verify the digest after the runtime starts with the imported weights. A tag alone cannot establish identity.

Add a profile to the existing `models` array in the manager's `models.json`.

```json
{"name":"d3-review","provider":"ollama","base_url":"http://ollama:11434","model":"d3:4b","role":"decision","digest":"<full verified Ollama digest>"}
```

Start or recreate the manager with `docker compose up -d --build --force-recreate models-configurator` and obtain the full decision profile ID from `GET /v1/models`. To use Laya, add a separate entry with model `laya:322m-multilingual-bf16` and its exact digest.

## Request a decision

Send one choice question and include the expected artifact digest as 64 lowercase hexadecimal characters. Use the immutable ID from the manager inventory for `model`.

```json
{"model":"<manager profile ID>","digest":"<verified artifact digest>","state":"Администратор может изменить лимит платежа.","question":{"type":"choice","instructions":"What does the text describe?","criteria":{"a":"A permission","b":"A response time"}}}
```

A successful response includes the profile ID, digest, answer and token usage. An input rejected by a measured model guard instead returns the same identity with a `reason` such as `input_limit`, `reserved_token` or `outside_training_context`. Platform or provider failures use an HTTP error. Decision calls share the manager's inference queue. The distribution is a model output and needs review before it is used as a decision.
