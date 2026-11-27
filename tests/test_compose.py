"""Check the deployed configuration without starting providers."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DeploymentTest(unittest.TestCase):
    def compose(self, *options, env=None):
        output = subprocess.check_output(["docker", "compose", *options, "config", "--format", "json"], env=env)
        return json.loads(output)

    def test_only_manager_runs_by_default(self):
        config = self.compose()
        self.assertEqual(set(config["services"]), {"models-configurator"})
        service = config["services"]["models-configurator"]
        self.assertEqual(service["ports"][0]["host_ip"], "127.0.0.1")
        self.assertEqual(service["ports"][0]["target"], 8317)
        self.assertEqual(config["networks"]["clients"]["name"], "models-configurator-clients")
        self.assertTrue(service["volumes"][0]["read_only"])

    def test_history_is_stored_in_a_dedicated_writable_volume(self):
        config = self.compose()
        service = config["services"]["models-configurator"]
        path = service["environment"]["MODELS_HISTORY_PATH"]
        volume = next(v for v in service["volumes"] if v["type"] == "volume")
        self.assertTrue(Path(path).is_relative_to(volume["target"]))
        self.assertEqual(volume["source"], "model_history")
        self.assertFalse(volume.get("read_only", False))
        self.assertIn("model_history", config["volumes"])

    def test_local_provider_is_private_to_manager(self):
        config = self.compose("--profile", "cpu")
        provider = config["services"]["ollama"]
        self.assertFalse(provider.get("ports"))
        self.assertNotIn("clients", provider["networks"])
        self.assertEqual(provider["environment"]["OLLAMA_NO_CLOUD"], "1")
        self.assertIn("runtime", config["services"]["models-configurator"]["networks"])

    def test_optional_arc_provider_stays_on_the_private_runtime_network(self):
        env = {**os.environ, 'MODELS_OLLAMA_DRI_DEVICE': '/dev/dri/renderD129',
               'MODELS_OLLAMA_RENDER_GID': '4321'}
        config = self.compose('-f', 'compose.yml', '-f', 'compose.ollama-arc.yml',
                              '--profile', 'arc', env=env)
        services = config['services']
        ollama = services['ollama']
        self.assertEqual(set(ollama['networks']), {'runtime'})
        self.assertEqual(ollama['profiles'], ['arc'])
        self.assertFalse(ollama.get('ports'))
        self.assertEqual(ollama['environment']['OLLAMA_NO_CLOUD'], '1')
        self.assertEqual(ollama['environment']['OLLAMA_VULKAN'], '1')
        self.assertEqual(ollama['environment']['GGML_VK_VISIBLE_DEVICES'], '0')
        self.assertEqual(ollama['environment']['OLLAMA_NUM_PARALLEL'], '1')
        self.assertEqual(ollama['group_add'], ['4321'])
        self.assertEqual(ollama['devices'], [{'source': '/dev/dri/renderD129',
                                              'target': '/dev/dri/renderD129', 'permissions': 'rw'}])
        self.assertEqual(ollama['volumes'][0]['source'], 'ollama_data')
        self.assertFalse(services['models-configurator'].get('devices'))

    def gpu_options(self, profile):
        return ('--env-file', '/dev/null', '-f', 'compose.yml',
                '-f', f'compose.ollama-{profile}.yml', '--profile', profile)

    def gpu_provider(self, profile, **settings):
        config = self.compose(*self.gpu_options(profile), env={**os.environ, **settings})
        services = config['services']
        ollama = services['ollama']
        self.assertEqual(ollama['profiles'], [profile])
        self.assertEqual(set(ollama['networks']), {'runtime'})
        self.assertFalse(ollama.get('ports'))
        self.assertFalse(ollama.get('privileged', False))
        self.assertEqual(ollama['environment']['OLLAMA_NO_CLOUD'], '1')
        self.assertEqual(ollama['environment']['OLLAMA_NUM_PARALLEL'], '1')
        self.assertEqual(ollama['environment']['OLLAMA_MAX_LOADED_MODELS'], '1')
        self.assertEqual(ollama['volumes'][0]['source'], 'ollama_data')
        self.assertEqual(ollama['volumes'][0]['target'], '/root/.ollama')
        self.assertFalse(services['models-configurator'].get('devices'))
        self.assertFalse(services['models-configurator'].get('deploy'))
        return ollama

    def test_nvidia_selects_one_gpu_by_uuid(self):
        uuid = 'GPU-12345678-1234-1234-1234-123456789abc'
        ollama = self.gpu_provider('nvidia', MODELS_OLLAMA_GPU_UUID=uuid)
        self.assertRegex(ollama['image'], r'^ollama/ollama:0\.40\.16@sha256:[a-f0-9]{64}$')
        self.assertFalse(ollama.get('devices'))
        reservations = ollama['deploy']['resources']['reservations']['devices']
        self.assertEqual(reservations, [{'driver': 'nvidia', 'device_ids': [uuid], 'capabilities': ['gpu']}])
        self.assertEqual(ollama['environment']['OLLAMA_VULKAN'], '0')

    def test_rocm_passes_only_the_selected_render_node_and_kfd(self):
        uuid = 'GPU-0123456789abcdef'
        node = '/dev/dri/renderD129'
        ollama = self.gpu_provider('rocm', MODELS_OLLAMA_GPU_UUID=uuid,
                                   MODELS_OLLAMA_DRI_DEVICE=node,
                                   MODELS_OLLAMA_RENDER_GID='4321', MODELS_OLLAMA_KFD_GID='4322')
        self.assertRegex(ollama['image'], r'^ollama/ollama:0\.40\.16-rocm@sha256:[a-f0-9]{64}$')
        self.assertEqual(ollama['devices'], [
            {'source': '/dev/kfd', 'target': '/dev/kfd', 'permissions': 'rw'},
            {'source': node, 'target': node, 'permissions': 'rw'},
        ])
        self.assertEqual(ollama['group_add'], ['4321', '4322'])
        self.assertEqual(ollama['environment']['OLLAMA_VULKAN'], '0')
        self.assertEqual(ollama['environment']['OLLAMA_IGPU_ENABLE'], '1')
        self.assertEqual(ollama['environment']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(ollama['environment']['ROCR_VISIBLE_DEVICES'], uuid)

    def test_amd_vulkan_passes_one_render_node_with_the_radv_icd(self):
        node = '/dev/dri/renderD130'
        ollama = self.gpu_provider('amd-vulkan', MODELS_OLLAMA_DRI_DEVICE=node,
                                   MODELS_OLLAMA_RENDER_GID='4321')
        self.assertEqual(ollama['build']['dockerfile'], 'Dockerfile.ollama-amd-vulkan')
        self.assertEqual(ollama['devices'], [{'source': node, 'target': node, 'permissions': 'rw'}])
        self.assertEqual(ollama['group_add'], ['4321'])
        env = ollama['environment']
        self.assertEqual(env['OLLAMA_VULKAN'], '1')
        self.assertEqual(env['OLLAMA_IGPU_ENABLE'], '1')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(env['ROCR_VISIBLE_DEVICES'], '-1')
        self.assertEqual(env['VK_DRIVER_FILES'], '/etc/models-configurator-amd-vulkan.json')
        self.assertEqual(env['VK_ICD_FILENAMES'], env['VK_DRIVER_FILES'])
        self.assertEqual(env['GGML_VK_VISIBLE_DEVICES'], '0')

    def test_new_gpu_profiles_require_explicit_device_settings(self):
        settings = {
            'MODELS_OLLAMA_GPU_UUID': 'GPU-0123456789abcdef',
            'MODELS_OLLAMA_DRI_DEVICE': '/dev/dri/renderD129',
            'MODELS_OLLAMA_RENDER_GID': '4321',
            'MODELS_OLLAMA_KFD_GID': '4322',
        }
        for profile, required in (
            ('nvidia', ('MODELS_OLLAMA_GPU_UUID',)),
            ('rocm', tuple(settings)),
            ('amd-vulkan', ('MODELS_OLLAMA_DRI_DEVICE', 'MODELS_OLLAMA_RENDER_GID')),
        ):
            for key in required:
                with self.subTest(profile=profile, missing=key):
                    env = {**os.environ, **settings, key: ''}
                    result = subprocess.run(
                        ['docker', 'compose', *self.gpu_options(profile), 'config'],
                        env=env, capture_output=True, text=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(key, result.stderr)

    def test_manager_parallel_request_limit_from_env_file(self):
        key = "MODELS_MAX_PARALLEL_REQUESTS"
        env = os.environ.copy()
        env.pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            for content, expected in [("", "1"), (f"{key}=3\n", "3"), (f"{key}=\n", "")]:
                with self.subTest(expected=expected):
                    dotenv.write_text(content)
                    config = self.compose("--env-file", str(dotenv), env=env)
                    manager = config["services"]["models-configurator"]
                    self.assertEqual(manager.get("environment", {}).get(key), expected)


if __name__ == "__main__":
    unittest.main()
