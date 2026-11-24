import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from runtime.bge import engine, prepare


class ProfileTests(unittest.TestCase):
    def test_reproducible_profile_binds_artifacts_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = b'fixture'
            digest = hashlib.sha256(data).hexdigest()
            pins = {name: digest for name in engine.ENCODER_HASHES}
            with patch.dict(engine.ENCODER_HASHES, pins):
                for name in [*pins, 'heads.npz']:
                    (base / name).write_bytes(data)
                pin = prepare.write_profile(base)
                self.assertEqual(engine.verify_profile(base, 'sha256:' + pin), pin)
                (base / 'heads.npz').write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    engine.verify_profile(base, pin)
                (base / 'heads.npz').write_bytes(data)
                profile = json.loads((base / 'profile.json').read_bytes())
                profile['rich_tokens'] = 999
                body = json.dumps(profile).encode()
                (base / 'profile.json').write_bytes(body)
                with self.assertRaises(ValueError):
                    engine.verify_profile(base, hashlib.sha256(body).hexdigest())

    def test_artifacts_reject_mismatched_hash_and_symbolic_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'real').write_bytes(b'data')
            (base / 'link').symlink_to(base / 'real')
            digest = hashlib.sha256(b'data').hexdigest()
            with self.assertRaises(ValueError):
                engine.check_file(base / 'link', digest)
            with self.assertRaises(ValueError):
                engine.check_file(base / 'real', '0' * 64)

    def test_failed_preparation_does_not_publish_partial_installation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepare.prepare(base, base / 'model', base, False)
            self.assertEqual(list(base.iterdir()), [])


class EngineTests(unittest.TestCase):
    def test_gpu_only(self):
        for device in ('AUTO', 'CPU', 'MULTI:GPU,CPU', 'GPUoops'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                engine.Engine(Path('/unused'), '0' * 64, device=device)

    def test_model_profile_dense_token_ceiling(self):
        self.assertEqual(engine.profile_spec({})['dense_tokens'], 512)

    def test_sparse_max_and_specials(self):
        self.assertEqual(engine.lexical_weights(
            [0, 3, 25, 25, 25, 30, 2, 1], [8, 8, 0.3, 0.5, 0.1, 0, 8, 8]),
            {'25': 0.5})

    def test_batch_and_token_limits_before_inference(self):
        runner = object.__new__(engine.Engine)
        class Tokenizer:
            def encode(self, text):
                class Tokens:
                    ids = [0] + [4] * int(text) + [2]
                return Tokens()
        runner._tokenizer = Tokenizer()
        for texts, modes in [(['127'], ['sparse']), (['511'], ['dense']),
                             (['1'] * 5, ['colbert']), (['1', '127'], ['dense', 'sparse'])]:
            with self.subTest(texts=texts), self.assertRaises(engine.InputLimitError):
                runner.encode(texts, modes)
        for texts, modes in [([], ['dense']), (['1'], []), (['1'], ['other']),
                             ([42], ['dense']), (['1'], ['dense', 'dense'])]:
            with self.assertRaises(ValueError):
                runner.encode(texts, modes)

    @unittest.skipUnless(importlib.util.find_spec('numpy'), 'requires numpy')
    def test_order_dense_slicing_and_colbert_eos(self):
        import numpy as np
        from types import SimpleNamespace
        runner = object.__new__(engine.Engine)
        runner._lock = threading.Lock()
        runner._tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[0, int(text), 2]))
        seen = []
        def encoder(inputs):
            self.assertEqual(inputs['attention_mask'].tolist(), [[1, 1, 1]])
            return {'last_hidden_state': inputs['input_ids'][..., None]}
        def heads(inputs):
            hidden = inputs[0].astype(float)
            seen.append(hidden.shape)
            return {'dense': hidden[:, 0], 'sparse': hidden, 'colbert': hidden}
        runner._encoder = encoder
        runner._heads = heads
        self.assertEqual(runner.encode(['4', '5'], ['colbert', 'sparse']), [
            {'colbert': [[4.0], [2.0]], 'sparse': {'4': 4.0}},
            {'colbert': [[5.0], [2.0]], 'sparse': {'5': 5.0}}])
        self.assertEqual(runner.encode(['4'], ['dense']), [{'dense': [0.0]}])
        self.assertEqual(seen, [(1, 3, 1), (1, 3, 1), (1, 1, 1)])

    @unittest.skipUnless(importlib.util.find_spec('openvino'), 'requires OpenVINO')
    def test_gpu_graph_math_against_reference_with_tiny_cpu_fixture(self):
        import numpy as np
        import openvino as ov
        hidden = np.array([[[3, 4], [2, -1], [0, 2]]], dtype=np.float32)
        arrays = dict(sparse_weight=np.array([[1, 1]], np.float32),
                      sparse_bias=np.array([-1], np.float32),
                      colbert_weight=np.array([[1, 0], [1, 1]], np.float32),
                      colbert_bias=np.array([0, 1], np.float32))
        model = ov.Core().compile_model(engine.projection_graph(arrays), 'CPU')
        results = model([hidden])
        np.testing.assert_allclose(results['dense'], [[0.6, 0.8]], atol=1e-5)
        np.testing.assert_allclose(results['sparse'], [[[6], [0], [1]]], atol=1e-5)
        projected = hidden @ arrays['colbert_weight'].T + arrays['colbert_bias']
        projected /= np.linalg.norm(projected, axis=-1, keepdims=True)
        np.testing.assert_allclose(results['colbert'], projected, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
