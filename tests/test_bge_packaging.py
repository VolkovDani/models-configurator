"""Render the BGE overlay alongside both NPU overlays without starting services."""
import json
from pathlib import Path
import os
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BgePackagingTest(unittest.TestCase):
    def test_private_gpu_sidecar_and_combined_socket_groups(self):
        settings = dict(MODELS_NPU_SOCKET_DIR="/npu-run", MODELS_NPU_SOCKET_GID="2001",
                        MODELS_NPU_MODEL_DIR="/npu", MODELS_NPU_MANIFEST="/manifest",
                        MODELS_NPU_PROFILE="a"*64, MODELS_NPU_UID="2011", MODELS_NPU_DEVICE_GID="2012",
                        MODELS_NPU_CACHE_DIR="/cache", MODELS_NPU_LOG_DIR="/logs",
                        MODELS_BGE_SOCKET_DIR="/bge-run", MODELS_BGE_SOCKET_GID="2002",
                        MODELS_BGE_MODEL_DIR="/bge", MODELS_BGE_PROFILE="b"*64,
                        MODELS_BGE_UID="2021", MODELS_BGE_DEVICE_GID="2022")
        files = ["compose.yml", "compose.npu-host.yml", "compose.npu.yml", "compose.bge-arc.yml"]
        cmd = ["docker", "compose", *(arg for file in files for arg in ("-f", file)),
               "config", "--format", "json"]
        services = json.loads(subprocess.check_output(cmd, cwd=ROOT, env={**os.environ, **settings}))["services"]
        manager, worker = services["models-configurator"], services["bge-runtime"]
        self.assertEqual(set(manager["group_add"]), {"2001", "2002"})
        self.assertFalse(manager.get("devices"))
        self.assertEqual(worker["network_mode"], "none")
        self.assertEqual(worker["user"], "2021:2002")
        self.assertEqual(worker["group_add"], ["2022"])
        self.assertEqual(worker["devices"], [{"source": "/dev/dri/renderD128", "target": "/dev/dri/renderD128", "permissions": "rw"}])
        self.assertTrue(worker["read_only"])
        self.assertEqual(worker["cap_drop"], ["ALL"])
        self.assertIn("no-new-privileges:true", worker["security_opt"])
        self.assertFalse(worker.get("env_file"))
        for field in ("ports", "privileged", "secrets"):
            self.assertFalse(worker.get(field))
        mounts = {v["source"]: v for v in worker["volumes"]}
        self.assertEqual(set(mounts), {"/bge", "/bge-run"})
        self.assertTrue(mounts["/bge"]["read_only"])
        self.assertFalse(mounts["/bge-run"].get("read_only"))
        self.assertIn("b"*64, worker["command"])
        for v in mounts.values():
            self.assertFalse(v.get("bind", {}).get("create_host_path"))
        client = {v["source"]: v for v in manager["volumes"]}
        self.assertTrue(client["/bge-run"]["read_only"])
        self.assertEqual(client["/bge-run"]["target"], "/run/models-bge")
        self.assertNotIn("/bge", client)
