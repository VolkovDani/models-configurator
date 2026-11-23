"""Check the deployed same-host boundary without starting providers."""
import json
import subprocess
import unittest


class DeploymentTest(unittest.TestCase):
    def compose(self, *profiles):
        flags = [arg for profile in profiles for arg in ("--profile", profile)]
        output = subprocess.check_output(["docker", "compose", *flags, "config", "--format", "json"])
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
        config = self.compose("cpu")
        provider = config["services"]["ollama"]
        self.assertFalse(provider.get("ports"))
        self.assertNotIn("clients", provider["networks"])
        self.assertEqual(provider["environment"]["OLLAMA_NO_CLOUD"], "1")
        self.assertIn("runtime", config["services"]["models-configurator"]["networks"])


if __name__ == "__main__":
    unittest.main()
