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
