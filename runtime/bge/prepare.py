"""Prepare a standalone, hash-pinned BGE-M3 installation from the verified FP16 export."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from . import engine


def write_profile(directory):
    files = {**engine.ENCODER_HASHES, 'heads.npz': engine.file_hash(directory / 'heads.npz')}
    path = directory / 'profile.json'
    path.write_text(json.dumps(engine.profile_spec(files), sort_keys=True, indent=2) + '\n',
                    encoding='utf-8')
    return engine.file_hash(path)


def convert_heads(directory):
    import numpy as np
    import torch

    arrays = {}
    for name, width in [('sparse', 1), ('colbert', 1024)]:
        path = directory / (name + '_linear.pt')
        engine.check_file(path, engine.HEAD_HASHES[path.name])
        state = torch.load(path, map_location='cpu', weights_only=True)
        for key, shape in [('weight', (width, 1024)), ('bias', (width,))]:
            tensor = state[key]
            if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                raise ValueError(f'unexpected {path.name} {key}')
            value = tensor.float().numpy()
            if not np.isfinite(value).all():
                raise ValueError(f'non-finite {path.name} {key}')
            arrays[f'{name}_{key}'] = value
        path.unlink()
    np.savez(directory / 'heads.npz', **arrays)


def prepare(encoder_dir, output_dir, heads_dir, download_heads):
    if (heads_dir is None) == (not download_heads):
        raise ValueError('select a local head directory or explicit download')
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-bge-', dir=output_dir.parent) as temp:
        work = Path(temp) / 'model'
        work.mkdir()
        for name, digest in engine.ENCODER_HASHES.items():
            target = work / name
            shutil.copyfile(encoder_dir / name, target)
            engine.check_file(target, digest)
        for name in engine.HEAD_HASHES:
            target = work / name
            if download_heads:
                url = f'https://huggingface.co/BAAI/bge-m3/resolve/{engine.REVISION}/{name}'
                with urllib.request.urlopen(url, timeout=60) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
            else:
                shutil.copyfile(heads_dir / name, target)
        convert_heads(work)
        pin = write_profile(work)
        for artifact in work.iterdir():
            artifact.chmod(0o444)
        work.rename(output_dir)
    return pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    heads = parser.add_mutually_exclusive_group(required=True)
    heads.add_argument('--heads-dir', type=Path)
    heads.add_argument('--download-heads', action='store_true')
    args = parser.parse_args()
    print(prepare(args.encoder_dir, args.output_dir, args.heads_dir, args.download_heads))


if __name__ == '__main__':
    main()
