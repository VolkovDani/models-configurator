"""Exercise the real Unix HTTP transport using a deterministic local engine."""
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import unittest

from runtime.bge import server


class FakeEngine:
    profile = "a" * 64
    device = "GPU.0"

    def __init__(self):
        self.calls = []

    def encode(self, inputs, modes):
        self.calls.append((inputs, modes))
        return [{mode: [] for mode in modes} for _ in inputs]


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "runtime.sock")
        self.engine = FakeEngine()
        self.service = server.Server(self.path, self.engine)
        self.addCleanup(self.service.server_close)
        thread = threading.Thread(target=self.service.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.service.shutdown)

    def request(self, method="POST", path="/v1/encode", body=None, headers=None):
        conn = HTTPConnection("localhost", timeout=3)
        try:
            conn.sock = socket.socket(socket.AF_UNIX)
            conn.sock.settimeout(3)
            conn.sock.connect(str(self.path))
            conn.request(method, path, body=body if isinstance(body, bytes) else json.dumps(body), headers=headers or {})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def payload(self, **fields):
        return {"profile": self.engine.profile, "input": ["local text"], "modes": ["dense"], **fields}

    def test_identity_and_requested_modes(self):
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o660)
        status = {"profile": self.engine.profile, "device": "GPU.0"}
        self.assertEqual(self.request("GET", "/status"), (200, {**status, "loaded": True}))
        modes = ["colbert", "sparse"]
        self.assertEqual(self.request(body=self.payload(modes=modes)),
                         (200, {**status, "data": [{"colbert": [], "sparse": []}]}))
        self.assertEqual(self.engine.calls, [(["local text"], modes)])

    def test_reject_invalid_before_inference(self):
        for body, expected in [
            (self.payload(profile="b" * 64), 409),
            (self.payload(extra=True), 422), (self.payload(modes=[]), 422),
            (self.payload(modes=["other"]), 422), (self.payload(modes=["dense", "dense"]), 422),
            (self.payload(input="text"), 422), (self.payload(input=[" "]), 422),
            (self.payload(input=["a"] * 33), 413), (self.payload(input=["a"] * 5, modes=["sparse"]), 413),
            (self.payload(input=["a" * 10001]), 413),
            (b'{"input": [], "input": [], "profile":"x","modes":["dense"]}', 422),
        ]:
            with self.subTest(expected=expected):
                self.assertEqual(self.request(body=body)[0], expected)
        self.assertEqual(self.request(body=b"", headers={"Content-Length": str((1 << 20) + 1)})[0], 413)
        self.assertFalse(self.engine.calls)

    def test_token_limit_and_internal_failures_do_not_disclose_text(self):
        for exc, code in [(server.InputLimitError("private input"), 413), (RuntimeError("private input"), 500),
                          (OSError("private input"), 500)]:
            def fail(*_):
                raise exc
            self.engine.encode = fail
            status, result = self.request(body=self.payload())
            self.assertEqual(status, code)
            self.assertNotIn("private input", str(result))

    def test_busy_never_dispatches_second_call_and_status_stays_available(self):
        entered, release = threading.Event(), threading.Event()
        encode = self.engine.encode
        def hold(inputs, modes):
            entered.set()
            if not release.wait(3):
                raise RuntimeError("test timed out")
            return encode(inputs, modes)
        self.engine.encode = hold
        first = []
        thread = threading.Thread(target=lambda: first.append(self.request(body=self.payload())))
        thread.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.request(body=self.payload())[0], 503)
            self.assertEqual(self.request("GET", "/status")[0], 200)
        finally:
            release.set()
            thread.join()
        self.assertEqual(first[0][0], 200)
        self.assertEqual(len(self.engine.calls), 1)

    def test_existing_socket_and_replacement_survive_close(self):
        with self.assertRaises(OSError):
            server.Server(self.path, self.engine)
        self.assertTrue(self.path.is_socket())
        os.unlink(self.path)
        self.path.write_text("keep")
        self.service.server_close()
        self.assertEqual(self.path.read_text(), "keep")
