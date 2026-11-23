"""Explicit download and offline export of the reviewed Qwen2.5 7B snapshot."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen

import npu_runtime


MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
FILES = {
    "LICENSE": "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e",
    "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
    "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model-00001-of-00004.safetensors": "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
    "model-00002-of-00004.safetensors": "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
    "model-00003-of-00004.safetensors": "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
    "model-00004-of-00004.safetensors": "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
EXPORT_OPTIONS = ["--task", "text-generation-with-past", "--weight-format", "int4",
                  "--sym", "--ratio", "1.0", "--group-size", "-1"]


def verify_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path.name}")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"unexpected digest for {path.name}")


def verify_source(source):
    if source.is_symlink() or not source.is_dir():
        raise ValueError("expected a local source directory")
    if {p.name for p in source.iterdir()} != FILES.keys():
        raise ValueError("source inventory contains missing or unexpected files")
    for name, expected in FILES.items():
        verify_file(source / name, expected)


def download(source):
    source.mkdir(mode=0o700, parents=True, exist_ok=True)
    if source.is_symlink() or any(p.name not in FILES for p in source.iterdir()):
        raise ValueError("choose a dedicated source directory without unexpected files")
    for name, expected in FILES.items():
        target = source / name
        if target.exists() or target.is_symlink():
            verify_file(target, expected)
            continue
        print(f"Downloading {name}", flush=True)
        # Publish only complete, verified files. A retry reuses previous shards.
        with tempfile.NamedTemporaryFile(dir=source.parent, prefix=".qwen-download-") as temp:
            with urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}", timeout=120) as response:
                shutil.copyfileobj(response, temp, length=1024 * 1024)
            temp.flush()
            verify_file(Path(temp.name), expected)
            os.link(temp.name, target)


def export_model(source, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    verify_source(source)
    source = source.resolve()
    output = output.absolute()
    if output.resolve().is_relative_to(source):
        raise ValueError("select an output directory outside the source")
    output.mkdir(mode=0o700)
    model = output / "model"
    with tempfile.TemporaryDirectory(dir=output) as work:
        env = dict(os.environ, TMPDIR=work, HF_HOME=work, HF_HUB_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1")
        subprocess.run(["optimum-cli", "export", "openvino", "--model", str(source),
                        *EXPORT_OPTIONS, str(model)], env=env, check=True)
    shutil.copyfile(source / "LICENSE", model / "SOURCE-LICENSE")
    (model / "export-freeze.txt").write_bytes(subprocess.check_output(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "freeze", "--all"]))
    recipe = "optimum-cli export openvino --model <verified-local-snapshot> " + " ".join(EXPORT_OPTIONS)
    manifest = npu_runtime.create_manifest(model, MODEL, REVISION, "Apache-2.0", recipe, 2048, 512)
    (output / "profile.json").write_bytes(npu_runtime.canonical(manifest) + b"\n")
    profile = npu_runtime.profile_id(manifest)
    (output / "profile.sha256").write_text(profile + "\n")
    print(profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="command", required=True)
    stages.add_parser("download", help="fetch the pinned public files from Hugging Face").add_argument("source", type=Path)
    export = stages.add_parser("export", help="verify source and generate a new local bundle (inside the export image)")
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "download":
            download(args.source)
        else:
            export_model(args.source, args.output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
