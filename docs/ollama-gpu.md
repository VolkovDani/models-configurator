# Ollama GPU profiles

[Русская версия](ollama-gpu.ru.md)

Use native Linux and Docker Compose 2.24.4 or later. Choose one Ollama overlay per installation. All four profiles use the existing `ollama` service name and `ollama_data` volume. The manager reaches Ollama at `http://ollama:11434` on its private runtime network. Client applications use the manager API.

| GPU | Overlay | Profile | Device selection |
| --- | --- | --- | --- |
| Intel Arc | `compose.ollama-arc.yml` | `arc` | Intel render node and its numeric group |
| NVIDIA | `compose.ollama-nvidia.yml` | `nvidia` | One NVIDIA UUID from `nvidia-smi -L`. Requires NVIDIA Container Toolkit. |
| AMD ROCm | `compose.ollama-rocm.yml` | `rocm` | One ROCm GPU UUID from `rocminfo`, the corresponding AMD render node, and the numeric groups of the node and `/dev/kfd`. Requires an Ollama-supported ROCm GPU. The pinned image is Linux amd64. |
| AMD Vulkan | `compose.ollama-amd-vulkan.yml` | `amd-vulkan` | AMD render node and its numeric group. Uses the RADV Vulkan driver. |

## Start Ollama

For a render-node profile, identify the GPU by PCI address with `lspci -Dnn` and `ls -l /dev/dri/by-path/*-render`. Use the address of that GPU to resolve the render node. For ROCm, match the UUID and PCI device reported by ROCm to the same adapter.

```bash
pci=0000:00:02.0 # Replace with the selected adapter's PCI address.
export MODELS_OLLAMA_DRI_DEVICE=$(readlink -e "/dev/dri/by-path/pci-${pci}-render")
export MODELS_OLLAMA_RENDER_GID=$(stat -c %g "$MODELS_OLLAMA_DRI_DEVICE")
```

For NVIDIA or ROCm, set `MODELS_OLLAMA_GPU_UUID` to the full `GPU-…` value for the selected adapter. For ROCm, also set `MODELS_OLLAMA_KFD_GID` to `$(stat -c %g /dev/kfd)`. The Vulkan profiles expose one render node with its vendor's Vulkan driver and select Vulkan index zero inside the container.

Set the profile and include its file after `compose.yml`. The example uses the base file and one Ollama overlay. Keep any settings, NPU, or BGE overlays already used by the installation. Save the Compose and device variables in the manager's private `.env` if Project Creator's linked CLI will start the manager.

```bash
export COMPOSE_PROFILES=arc # Or nvidia, rocm, amd-vulkan.
export COMPOSE_FILE="compose.yml:compose.ollama-${COMPOSE_PROFILES}.yml"
docker compose config --quiet
docker compose up -d --build ollama
```

Follow [Add models](../README.md#add-models) to pin every enabled role to its exact Ollama tag and full manifest SHA-256. Use the weights already installed in the manager volume, pull a known model, or restore a local backup of the legacy weights into the manager's own volume. Keep the legacy volume for rollback. Verify each manifest digest after pulling or restoring and before configuring clients. The manager checks the Ollama model digest before and after inference.

## Verify hardware and memory placement

Start Ollama freshly on the selected profile. Perform one synthetic request per model and role through the manager API. For Project Creator, verify structured chat and 1024-component embeddings as applicable, and run the decision probe if that role is in use. Use the same model digests, input and request options on the old and new endpoints. Inspect placement immediately after each request, before another model displaces it. [API examples](../README.md#connect-applications) · [Decision probe](decision-models.md#request-a-decision)

Set `model` to the exact Ollama tag and `digest` to its full configured digest. This command reads the private Ollama API from a temporary container in the same network namespace. Run it from Bash with `jq` installed.

```bash
set -o pipefail
ollama_container=$(docker compose ps -q ollama)
docker run --rm --network "container:$ollama_container" \
  alpine:3.22@sha256:49340b346798875b5592e5d420241f2f9cfba5f47e62ebb1446a1b34734e898f \
  wget -qO- http://127.0.0.1:11434/api/ps |
  jq -e --arg model "$model" --arg digest "${digest#sha256:}" '
    .models[] | select(.name == $model and (.digest | ltrimstr("sha256:")) == $digest
      and (.remote_host // "") == "" and (.remote_model // "") == "")
    | select((.size | type) == "number" and (.size_vram | type) == "number"
      and .size_vram > 0 and .size_vram <= .size)
    | {name, digest, size, size_vram}'
docker compose logs --no-color ollama
```

Record the request result, timestamp, Ollama image and version, model digest, GPU name and PCI address or UUID, and both reported byte counts. Match Ollama's `inference compute` entry for the active backend and GPU to the selected adapter, and check the model load/offload entries. `size_vram == size` indicates full reported GPU placement. A smaller positive value indicates split CPU/GPU placement. On integrated GPUs the reported GPU allocation uses shared system memory. A CPU-only result, missing or mismatched digest, wrong adapter, or software Vulkan renderer fails the GPU check.

## Verification results

On 2026-11-27, the Project Creator Ollama endpoint and the manager API both ran synthetic structured chat and embeddings on the Intel Arc iGPU in a Core Ultra 7 268V (`8086:64a0`, PCI `0000:00:02.0`, `/dev/dri/renderD128`). Both ran Ollama 0.40.16 with Vulkan. The legacy image used Mesa `25.2.8-0ubuntu0.24.04.4` and the manager image used `25.2.8-0ubuntu0.24.04.5`.

| Model | Ollama digest | `size_vram` and `size` on both endpoints |
| --- | --- | ---: |
| `qwen2.5:0.5b` | `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` | 456,361,245 bytes each |
| `bge-m3:567m` | `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` | 689,187,061 bytes each |

The JSON answers matched, all 1024 embedding components matched, and both runtimes reported all 25 layers offloaded for each model. This run exercised the model endpoints with synthetic inputs. Recheck the models and roles your installation uses. NVIDIA, AMD ROCm, and AMD Vulkan still require real inference and placement checks on their respective hardware. [Migration issue](https://github.com/VolkovDani/models-configurator/issues/3)

## Switch Project Creator after verification

After a profile passes on the target machine, connect each verified role to the manager using its full ID from `/v1/models`. Keep any role still using the bundled runtime, and retain the saved device selection and weights for rollback. Switch an installation to `--device remote --switch-device` only when all its active roles use the manager API. A change in the model profile ID can require rebuilding embeddings for that profile. Follow the Project Creator connection and rollback instructions. [Project Creator installation guide](https://github.com/VolkovDani/project-creator/blob/develop/docs/setup-cli.en.md) · [Managed model profiles](https://github.com/VolkovDani/project-creator/blob/develop/docs/adr/012-standalone-model-manager.md#connect-a-compose-installation)

To switch the manager back to CPU, remove its GPU overlay, replace its Ollama profile with `cpu`, and run `docker compose up -d --force-recreate ollama` under the same Compose project. Preserve the existing volume.

[Ollama GPU support](https://docs.ollama.com/gpu) · [Docker GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/) · [AMD ROCm device passthrough](https://rocm.docs.amd.com/en/latest/install/docker-containers.html)
