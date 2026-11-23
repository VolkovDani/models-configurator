import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import npu_runtime as npu


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / 'model'
        self.model.mkdir()
        for name in ('openvino_model.xml', 'openvino_model.bin',
                     'openvino_tokenizer.xml', 'openvino_detokenizer.xml'):
            (self.model / name).write_bytes(b'fixture')

    def manifest(self):
        return npu.create_manifest(self.model, 'Qwen/Qwen2.5-7B-Instruct', 'a'*40,
                                   'Apache-2.0', 'optimum-cli export openvino --local-files-only', 2048, 512)

    def test_profile_pins_all_files_and_generation_limits(self):
        first = self.manifest()
        self.assertEqual(npu.profile_id(first), npu.profile_id(json.loads(json.dumps(first))))
        changed = dict(first, max_prompt_tokens=1024)
        self.assertNotEqual(npu.profile_id(first), npu.profile_id(changed))
        (self.model / 'openvino_model.bin').write_bytes(b'changed')
        self.assertNotEqual(npu.profile_id(first), npu.profile_id(self.manifest()))
        with self.assertRaises(ValueError):
            npu.verify_model(self.model, first)

    def test_symlink_missing_and_unpinned_sources_rejected(self):
        (self.model / 'link').symlink_to(self.model / 'openvino_model.bin')
        with self.assertRaises(ValueError):
            self.manifest()
        (self.model / 'link').unlink()
        with self.assertRaises(ValueError):
            npu.create_manifest(self.model, 'Qwen/model', 'main', 'Apache-2.0', 'export', 2048, 512)
        (self.model / 'openvino_model.bin').unlink()
        with self.assertRaises(ValueError):
            self.manifest()

    def test_runtime_and_device_are_pinned(self):
        manifest = self.manifest()
        npu.validate_manifest(manifest)
        for key, value in [('device', 'AUTO'), ('max_prompt_tokens', True),
                           ('runtime', {'openvino': 'latest'})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                npu.validate_manifest(dict(manifest, **{key: value}))

    def test_profile_requires_schema_constrained_generation(self):
        manifest = self.manifest()
        self.assertEqual(manifest['output_format'], 'json-schema-v1')
        self.assertEqual(manifest['version'], 2)
        old = dict(manifest, version=1)
        old.pop('output_format')
        with self.assertRaises(ValueError):
            npu.validate_manifest(old)
        with self.assertRaises(ValueError):
            npu.validate_manifest(dict(manifest, output_format='prompt-only'))



def request():
    return {'profile': 'a'*64, 'messages': [
        {'role': 'system', 'content': 'Use the supplied units.'},
        {'role': 'user', 'content': 'Convert 8 meters.'}], 'schema': {'type': 'object'}}


CONTENT = ' { "value": 8, "unit": "m" } '


class ContractTests(unittest.TestCase):
    def test_validate_bounded_generic_messages_and_schema(self):
        npu.validate_request(request(), 'a'*64)
        npu.validate_request(dict(request(), max_completion_tokens=1024), 'a'*64)
        for overrides in (
            {'profile': 'b'*64}, {'messages': []}, {'messages': [request()['messages'][0]]*129},
            {'messages': [{'role': 'other', 'content': 'x'}]},
            {'messages': [{'role': 'system', 'content': ' '}]},
            {'messages': [{'role': 'user', 'content': 1}]},
            {'messages': [{'role': 'user', 'content': 'ok', 'extra': True}]},
            {'schema': None}, {'max_completion_tokens': True}, {'max_completion_tokens': 0},
            {'max_completion_tokens': 16385}, {'temperature': 0.5},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                npu.validate_request(dict(request(), **overrides), 'a'*64)

    def test_template_preserves_conversation_and_counts_all_tokens(self):
        payload = request()
        payload['messages'].extend([{'role': 'assistant', 'content': '8 m'},
                                    {'role': 'user', 'content': 'Convert to feet.'}])
        tokenizer = Mock()
        encoded = SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 101)))
        tokenizer.encode.return_value = encoded
        with self.assertRaises(ValueError):
            npu.format_prompt(tokenizer, payload, 100)
        self.assertEqual(npu.format_prompt(tokenizer, payload, 101), (encoded, 101))
        history = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(history[1:], payload['messages'][1:])
        self.assertEqual(history[0]['content'], 'Use the supplied units.\nReturn one JSON object '
                         'matching this schema, without markdown:\n{"type": "object"}')
        self.assertEqual(payload['messages'][0]['content'], 'Use the supplied units.')
        tokenizer.encode.assert_called_with(tokenizer.apply_chat_template.return_value,
                                             add_special_tokens=False, truncation=False)
        del payload['messages'][0]
        npu.format_prompt(tokenizer, payload, 101)
        history = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(history[0]['role'], 'system')
        self.assertEqual(history[1:], payload['messages'])


if __name__ == '__main__':
    unittest.main()
