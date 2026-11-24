"""Pinned BAAI/bge-m3 encoder and retrieval heads on an explicit OpenVINO GPU."""
import hashlib
import json
from pathlib import Path
import re
import threading

REVISION = '5617a9f61b028005a4858fdac845db406aefb181'
ENCODER_HASHES = {
    'openvino_model.bin': 'd44168f40277953a660a1e1c1fccbdbdf8a7b92aa7833aa3eb26ecf2351c9312',
    'openvino_model.xml': 'fc825c2c2c5aa6b721cd78f09d20e786fa2ca3e393ec873ade653a87845cc5b1',
    'tokenizer.json': 'bc552f2faa9ccaf0be4136806b77d442c6f81061a20ce3c569927d3908599284',
}
HEAD_HASHES = {
    'sparse_linear.pt': '45c93804d2142b8f6d7ec6914ae23a1eee9c6a1d27d83d908a20d2afb3595ad9',
    'colbert_linear.pt': '19bfbae397c2b7524158c919d0e9b19393c5639d098f0a66932c91ed8f5f9abb',
}
DENSE_TOKENS = 512
RICH_TOKENS = 128
DENSE_BATCH = 32
RICH_BATCH = 4
MODES = ('dense', 'sparse', 'colbert')


class InputLimitError(ValueError):
    """A representation's batch or token limit was exceeded."""


def profile_spec(files):
    return {
        'format': 1, 'model': 'BAAI/bge-m3', 'revision': REVISION, 'files': files,
        'head_sources': HEAD_HASHES, 'dimension': 1024,
        'tokenization': 'special tokens, no prefixes, truncation or padding',
        'dense': 'L2-normalized CLS', 'dense_tokens': DENSE_TOKENS, 'dense_batch': DENSE_BATCH,
        'sparse': 'ReLU(linear), positive max per ID, excluding CLS/PAD/EOS/UNK',
        'colbert': 'L2-normalized linear, omit CLS and padding, include EOS',
        'rich_tokens': RICH_TOKENS, 'rich_batch': RICH_BATCH,
    }


def file_hash(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'expected regular file: {path.name}')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_file(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f'wrong SHA-256: {path.name}')


def verify_profile(directory, expected):
    digest = expected.removeprefix('sha256:')
    if not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('invalid profile SHA-256')
    path = directory / 'profile.json'
    check_file(path, digest)
    profile = json.loads(path.read_bytes())
    if not isinstance(profile, dict) or not isinstance(profile.get('files'), dict):
        raise ValueError('invalid profile')
    files = profile['files']
    if (set(files) != {*ENCODER_HASHES, 'heads.npz'} or
            any(files[k] != v for k, v in ENCODER_HASHES.items()) or
            profile != profile_spec(files)):
        raise ValueError('unsupported artifact or preprocessing profile')
    for name, expected_file in files.items():
        check_file(directory / name, expected_file)
    return digest


def lexical_weights(ids, weights):
    result = {}
    for token, weight in zip(ids, weights, strict=True):
        # The pinned tokenizer uses 0, 1, 2, 3 for CLS, PAD, EOS and UNK.
        if token > 3 and weight > result.get(str(token), 0):
            result[str(token)] = float(weight)
    return result


def projection_graph(arrays):
    import numpy as np
    import openvino as ov
    from openvino import opset13 as op

    width = arrays['sparse_weight'].shape[1]
    hidden = op.parameter([1, -1, width], np.float32, name='hidden')

    def linear(name):
        weight = op.constant(arrays[name + '_weight'])
        return op.add(op.matmul(hidden, weight, False, True), op.constant(arrays[name + '_bias']))

    def normalize(node):
        # FlagEmbedding applies torch.nn.functional.normalize with eps=1e-12.
        return op.normalize_l2(node, op.constant(np.array([-1], np.int64)), 1e-24, 'max')

    cls = op.gather(hidden, op.constant(np.int64(0)), op.constant(np.int64(1)))
    outputs = {'dense': normalize(cls), 'sparse': op.relu(linear('sparse')),
               'colbert': normalize(linear('colbert'))}
    for name, node in outputs.items():
        node.output(0).get_tensor().set_names({name})
    return ov.Model(list(outputs.values()), [hidden], 'bge_m3_projections')


class Engine:
    def __init__(self, model_dir: Path, expected_profile: str, device='GPU'):
        if not re.fullmatch(r'GPU(?:\.[0-9]+)?', device):
            raise ValueError('an explicit OpenVINO GPU device is required')
        self.profile = verify_profile(model_dir, expected_profile)

        import numpy as np
        import openvino as ov
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        core = ov.Core()
        encoder = core.read_model(model_dir / 'openvino_model.xml')
        # Keep sequence length dynamic. An upper bound makes the Intel GPU plugin
        # reserve attention buffers for the longest sequence during compilation.
        encoder.reshape({i.any_name: [1, -1] for i in encoder.inputs})
        self._encoder = core.compile_model(encoder, device, {'PERFORMANCE_HINT': 'LATENCY'})
        with np.load(model_dir / 'heads.npz', allow_pickle=False) as arrays:
            self._heads = core.compile_model(projection_graph(arrays), device)
        devices = []
        for compiled in (self._encoder, self._heads):
            actual = list(compiled.get_property('EXECUTION_DEVICES'))
            if len(actual) != 1 or not re.fullmatch(r'GPU(?:\.[0-9]+)?', actual[0]):
                raise RuntimeError('model did not compile exclusively for the GPU')
            devices.append(actual[0])
        if devices[0] != devices[1]:
            raise RuntimeError('encoder and heads compiled for different GPUs')
        self.device = devices[0]
        self._lock = threading.Lock()

    def encode(self, texts: list[str], modes: list[str]):
        if (not isinstance(texts, list) or not texts or
                any(not isinstance(text, str) or not text.strip() for text in texts)):
            raise ValueError('expected a non-empty list of texts')
        if (not isinstance(modes, list) or not modes or any(mode not in MODES for mode in modes)
                or len(set(modes)) != len(modes)):
            raise ValueError('invalid representation list')
        rich = any(mode != 'dense' for mode in modes)
        if len(texts) > (RICH_BATCH if rich else DENSE_BATCH):
            raise InputLimitError('batch limit exceeded')
        rows = [self._tokenizer.encode(text).ids for text in texts]
        if any(len(row) > (RICH_TOKENS if rich else DENSE_TOKENS) for row in rows):
            raise InputLimitError('token limit exceeded')
        with self._lock:
            return [self._encode_row(row, modes, rich) for row in rows]

    def _encode_row(self, row, modes, rich):
        import numpy as np

        ids = np.array([row], np.int64)
        hidden = self._encoder({'input_ids': ids, 'attention_mask': np.ones_like(ids)})['last_hidden_state']
        # Dense-only calls need the projections of just the CLS token.
        outputs = self._heads([hidden if rich else hidden[:, :1, :]])
        result = {}
        for mode in modes:
            values = outputs[mode][0]
            if not np.isfinite(values).all():
                raise RuntimeError('model produced a non-finite representation')
            if mode == 'sparse':
                result[mode] = lexical_weights(row, values[:, 0])
            elif mode == 'colbert':
                result[mode] = values[1:].tolist()
            else:
                result[mode] = values.tolist()
        return result
