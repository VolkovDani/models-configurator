"""NPU deployment boundaries, checked without weights, drivers or hardware."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def compose(*overlays):
    settings = dict(MODELS_NPU_SOCKET_DIR="/example/run", MODELS_NPU_SOCKET_GID="1234",
                    MODELS_NPU_MODEL_DIR="/example/model", MODELS_NPU_MANIFEST="/example/manifest.json",
                    MODELS_NPU_PROFILE="d" * 64, MODELS_NPU_CACHE_DIR="/example/cache",
                    MODELS_NPU_LOG_DIR="/example/logs", MODELS_NPU_UID="1235", MODELS_NPU_DEVICE_GID="1236")
    files = [arg for file in ("compose.yml", *overlays) for arg in ("-f", file)]
    return json.loads(subprocess.check_output(["docker", "compose", *files, "config", "--format", "json"],
                                              cwd=ROOT, env={**os.environ, **settings}))["services"]


class PackagingTest(unittest.TestCase):
    def test_host_runtime_shares_only_socket_directory_with_manager(self):
        services = compose("compose.npu-host.yml")
        self.assertEqual(set(services), {"models-configurator"})
        manager = services["models-configurator"]
        self.assertFalse(manager.get("devices"))
        self.assertEqual(manager["group_add"], ["1234"])
        mounts = {v["target"]: v for v in manager["volumes"] if v["type"] == "bind"}
        self.assertEqual(set(mounts), {"/etc/models.json", "/run/models-npu"})
        self.assertTrue(mounts["/run/models-npu"]["read_only"])
        self.assertEqual(mounts["/run/models-npu"]["source"], "/example/run")
        self.assertFalse(mounts["/run/models-npu"].get("bind", {}).get("create_host_path"))

    def test_container_is_offline_nonroot_and_uses_explicit_read_only_artifacts(self):
        services = compose("compose.npu-host.yml", "compose.npu.yml")
        runtime = services["npu-runtime"]
        self.assertEqual(runtime["network_mode"], "none")
        self.assertFalse(runtime.get("privileged"))
        self.assertEqual(runtime["user"], "1235:1234")
        self.assertEqual(runtime["group_add"], ["1236"])
        self.assertEqual([d["source"] for d in runtime["devices"]], ["/dev/accel/accel0"])
        self.assertFalse(services["models-configurator"].get("devices"))
        self.assertIn("d" * 64, runtime["command"])
        mounts = {v["source"]: v for v in runtime["volumes"]}
        for source in ("/example/model", "/example/manifest.json"):
            self.assertTrue(mounts[source]["read_only"])
        for source in ("/example/run", "/example/cache", "/example/logs"):
            self.assertFalse(mounts[source].get("read_only"))
        for mount in mounts.values():
            self.assertFalse(mount.get("bind", {}).get("create_host_path"))

    def test_native_and_container_installs_use_the_same_hashed_pip_wheel(self):
        folder = ROOT / "runtime/npu"
        requirement = (folder / "requirements-installer.txt").read_text()
        self.assertRegex(requirement, r"https://files\.pythonhosted\.org/.+/pip-26\.3-py3-none-any\.whl#sha256=[a-f0-9]{64}")
        for name in ("Dockerfile", "Dockerfile.export"):
            self.assertIn("requirements-installer.txt", (folder / name).read_text())
        self.assertIn("requirements-installer.txt", (ROOT / "scripts/install-npu-runtime").read_text())

    def test_installer_preserves_an_existing_environment(self):
        with tempfile.TemporaryDirectory() as path:
            probe = Path(path, "keep")
            probe.write_text("intact")
            result = subprocess.run([ROOT / "scripts/install-npu-runtime", path], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already exists", result.stderr)
            self.assertEqual(probe.read_text(), "intact")

    def test_failed_install_preserves_directory_for_inspection(self):
        with tempfile.TemporaryDirectory() as path:
            target = Path(path, "env")
            result = subprocess.run([ROOT / "scripts/install-npu-runtime", "--python", "/bin/false", target],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(target.is_dir())
            self.assertIn("incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
