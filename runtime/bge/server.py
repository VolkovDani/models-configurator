"""Private Unix HTTP adapter for the offline BGE-M3 engine."""
import argparse
from http.server import BaseHTTPRequestHandler
import os
from pathlib import Path
import signal
import threading

from runtime.bge.engine import DENSE_BATCH, InputLimitError, MODES, RICH_BATCH
from runtime.npu.npu_runtime import UnixHTTPServer, canonical, strict_json


MAX_BODY = 1 << 20


class Server(UnixHTTPServer):
    def __init__(self, path, engine):
        self.engine = engine
        self.lock = threading.Lock()
        super().__init__(os.fspath(path), Handler)


class BadRequest(ValueError):
    def __init__(self, status):
        self.status = status


def parse_request(data, profile):
    try:
        request = strict_json(data.decode("utf-8"))
    except (ValueError, RecursionError):
        raise BadRequest(422) from None
    if not isinstance(request, dict) or request.keys() != {"profile", "input", "modes"}:
        raise BadRequest(422)
    if request["profile"] != profile:
        raise BadRequest(409)
    modes, inputs = request["modes"], request["input"]
    if (not isinstance(modes, list) or not 1 <= len(modes) <= len(MODES)
            or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes)):
        raise BadRequest(422)
    if not isinstance(inputs, list) or not inputs:
        raise BadRequest(422)
    if len(inputs) > (DENSE_BATCH if modes == ["dense"] else RICH_BATCH):
        raise BadRequest(413)
    for text in inputs:
        if not isinstance(text, str) or not text.strip():
            raise BadRequest(422)
        if len(text) > 10000:
            raise BadRequest(413)
        try:
            text.encode("utf-8")  # Reject lone surrogates before passing text to Rust.
        except UnicodeEncodeError:
            raise BadRequest(422) from None
    return inputs, modes


class Handler(BaseHTTPRequestHandler):
    timeout = 5

    def log_message(self, *_):
        pass  # Do not log text, vectors or request paths.

    def reply(self, status, value):
        data = canonical(value)
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass

    def do_GET(self):
        if self.path != "/status":
            self.reply(404, {"error": "unknown route"})
            return
        engine = self.server.engine
        self.reply(200, {"profile": engine.profile, "device": engine.device, "loaded": True})

    def do_POST(self):
        if self.path != "/v1/encode":
            self.reply(404, {"error": "unknown route"})
            return
        try:
            lengths = self.headers.get_all("Content-Length", [])
            if (self.headers.get_all("Transfer-Encoding") or len(lengths) != 1
                    or not lengths[0].isascii() or not lengths[0].isdigit()):
                raise BadRequest(422)
            if len(lengths[0]) > 10 or int(lengths[0]) > MAX_BODY:
                raise BadRequest(413)
            length = int(lengths[0])
            data = self.rfile.read(length)
            if len(data) != length:
                raise BadRequest(422)
            engine = self.server.engine
            inputs, modes = parse_request(data, engine.profile)
            if not self.server.lock.acquire(blocking=False):
                raise BadRequest(503)
            try:
                rows, input_tokens = engine.encode(inputs, modes)
            finally:
                self.server.lock.release()
            self.reply(200, {"profile": engine.profile, "device": engine.device,
                             "data": rows, "input_tokens": input_tokens})
        except InputLimitError:
            self.reply(413, {"error": "BGE input limit exceeded"})
        except BadRequest as error:
            self.reply(error.status, {"error": "BGE request rejected"})
        except Exception:
            self.reply(500, {"error": "BGE inference failed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--socket", required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() == 0:
        parser.error("run the BGE worker with a nonroot UID")
    if not args.socket.is_absolute() or len(os.fsencode(args.socket)) > 100:
        parser.error("use an absolute Unix socket path of at most 100 bytes")
    if os.path.lexists(args.socket):
        parser.error("socket path exists; stop the owner or remove a confirmed stale socket")
    from runtime.bge.engine import Engine
    engine = Engine(args.model, args.profile, device="GPU")

    def terminate(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    previous = os.umask(0o117)  # Restrict the socket even before chmod in server_bind.
    try:
        with Server(args.socket, engine) as service:
            os.umask(previous)
            service.serve_forever()
    finally:
        os.umask(previous)


if __name__ == "__main__":
    main()
