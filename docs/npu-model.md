# Prepare a new Qwen NPU model

[Русская версия](npu-model.ru.md) · [Runtime setup](npu.md)

Use this procedure on Linux x86_64 for a fresh installation. To reuse a verified local OpenVINO IR and its matching manifest, go directly to runtime setup. Model preparation requires Python 3.11+ on the host, Docker Engine, about 40–50 GB free on disk, and about 19 GiB of available RAM for export. Stop other memory-intensive jobs before exporting.

## Fetch the source and exporter

From the Models Configurator repository, download the public Apache-2.0 source to a dedicated private directory. This sends anonymous HTTPS requests for twelve named files at the pinned commit, including four Safetensors shards (15.23 GB total weights). Each file must match its expected SHA256. After an interrupted download, rerun the same command to reuse completed, verified shards. An existing mismatched file is preserved and reported. [Source snapshot](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/tree/a09a35458c702b33eeacc393d103063234e8bc28)

```bash
mkdir -p .npu logs/npu
python3 runtime/npu/prepare_qwen.py download "$PWD/.npu/qwen-source"
docker build -f runtime/npu/Dockerfile.export -t models-configurator-qwen-export:2.2.0 runtime/npu
```

The build downloads the separate CPython 3.12 CPU exporter and tests the torchvision native operator and Optimum Intel import. Its dependency versions come from the 2026-11-17 hardware-tested environment. The installer first upgrades pip to 26.3 using a hash-pinned official wheel. The CPU Torch and torchvision wheels also carry upstream hashes. As of 2026-11-23, PyPI's version-specific advisory records for the 60 pinned distributions reported two advisories for Transformers 5.5.4. Optimum Intel 2.2.0 requires Transformers below 5.6. Keep this toolchain confined to the offline container and the reviewed source below. [Tested toolchain](https://github.com/VolkovDani/project-creator/blob/4efe6c8/docs/dependency-updates.md) · [Tokenizer path advisory](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w) · [Custom generation advisory](https://github.com/advisories/GHSA-x9r9-c232-4q39)

## Export offline

Run as your regular Linux user. Use a disk-backed directory for the export because intermediate files alone can exceed 15 GB. The container has a read-only root, runs without network access or credentials, mounts only the verified source read-only, and writes its bundle and temporary model files under the dedicated output directory. The command requires a previously built local image. It rechecks every source hash and rejects extra files and symbolic links before invoking the exporter.

```bash
mkdir -p .npu/exports
set -o pipefail
docker run --rm --pull never --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD/.npu/qwen-source,target=/source,readonly" \
  --mount "type=bind,source=$PWD/.npu/exports,target=/out" \
  models-configurator-qwen-export:2.2.0 2>&1 | tee logs/npu/qwen-export.log
```

The recipe uses symmetric channel-wise INT4, ratio 1.0, group size -1 and the `text-generation-with-past` task. The completed `.npu/exports/qwen/` contains `model/` (IR, tokenizer, license and exporter freeze), `profile.json` (all local file hashes, source revision, recipe, runtime versions, 2048 input and 512 output token budgets) and `profile.sha256`. The profile identifies this specific export. The runtime rechecks the pin and file inventory before loading. An existing `qwen/` is preserved. After a failed export inspect its log, choose a new empty output parent and retry.

For container runtime setup, select the resulting files explicitly:

```bash
export MODELS_NPU_MODEL_DIR="$PWD/.npu/exports/qwen/model"
export MODELS_NPU_MANIFEST="$PWD/.npu/exports/qwen/profile.json"
export MODELS_NPU_PROFILE="$(cat .npu/exports/qwen/profile.sha256)"
```

The twelve source pins were checked against the upstream snapshot metadata and the full local cached source. Deterministic onboarding tests exercise download verification, offline export arguments, manifest generation and protection of existing files with small fixtures. The actual 7B export with this recipe and dependency stack was hardware-tested in Project Creator on 2026-11-17. The new manager exporter image was checked separately for dependency consistency and native imports. No second full export was run for this migration. [Hardware results](https://github.com/VolkovDani/project-creator/blob/4efe6c8/docs/dependency-updates.md)
