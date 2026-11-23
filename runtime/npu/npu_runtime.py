#!/usr/bin/env python3
"""Opt-in, offline NPU-only structured generation service. Standard library except worker."""
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import re
import select
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler


VERSIONS = {'openvino': '2026.4.1', 'openvino-genai': '2026.4.1.0',
            'openvino-tokenizers': '2026.4.1.0', 'numpy': '2.5.4',
            'openvino-telemetry': '2025.2.0'}
MAX_BODY = 65536
HEX = re.compile(r'[0-9a-f]{64}')
REQUIRED_FILES = {'openvino_model.xml', 'openvino_model.bin',
                  'openvino_tokenizer.xml', 'openvino_detokenizer.xml'}
LIFECYCLE_LOG = Path(__file__).resolve().parents[2] / 'logs/npu/runtime-lifecycle.jsonl'
CALL_ID = re.compile(r'[0-9a-f]{32}')
RUNTIME_ERROR_MESSAGES = {
    'NPU_UNAVAILABLE': 'NPU unavailable or request cancelled; explicit restart required',
    'NPU_RECOVERY_FAILED': 'NPU recovery failed',
    'NPU_WORKER_EXIT': 'NPU worker exited; explicit restart required',
    'TIMEOUT': 'NPU request timed out; restart runtime before retry',
    'INVALID_OUTPUT': 'NPU output rejected',
    'PROFILE_CHANGED': 'NPU model changed or profile differs; explicit restart required',
}
_LIFECYCLE_LOCK = threading.Lock()


class RuntimeFailure(ValueError):
    """Typed public failure with a fixed safe message and allowlisted code."""
    def __init__(self, code, *_ignored):
        if not isinstance(code, str) or code not in RUNTIME_ERROR_MESSAGES:
            raise ValueError('unknown NPU runtime error code')
        self.code = code
        super().__init__(RUNTIME_ERROR_MESSAGES[code])


def lifecycle_event(event, *, profile=None, stage=None, call_id=None, **_ignored):
    """Append fixed technical fields only; never persist request/model content."""
    record = {'time': time.time(), 'event': re.sub(r'[^a-z0-9_]', '_', str(event))[:64],
              'device': 'NPU'}
    if isinstance(profile, str) and HEX.fullmatch(profile):
        record['profile'] = profile
    if isinstance(stage, str) and re.fullmatch(r'[A-Z0-9_]{1,64}', stage):
        record['stage'] = stage
    if isinstance(call_id, str) and CALL_ID.fullmatch(call_id):
        record['call_id'] = call_id
    try:
        with _LIFECYCLE_LOCK:
            LIFECYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
            with LIFECYCLE_LOG.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, sort_keys=True) + '\n')
    except OSError:
        # Diagnostics must never change inference, cancellation, or recovery behavior.
        pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def profile_id(manifest):
    return hashlib.sha256(canonical(manifest)).hexdigest()


def inventory(model, hashes=True):
    model = Path(model)
    if model.is_symlink() or not model.is_dir():
        raise ValueError('model must be a real local directory')
    result = {}
    for path in sorted(model.rglob('*')):
        if path.is_symlink():
            raise ValueError('symlinks in model directory are forbidden')
        if path.is_dir():
            continue
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('model contains a non-regular file')
        name = path.relative_to(model).as_posix()
        if hashes:
            with path.open('rb') as stream:
                result[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        else:
            result[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if not REQUIRED_FILES <= result.keys():
        raise ValueError('incomplete OpenVINO model/tokenizer export')
    return result


def validate_manifest(value):
    expected = {'version', 'device', 'runtime', 'output_format', 'model_id', 'revision', 'license',
                'export_command', 'max_prompt_tokens', 'max_new_tokens', 'files'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('invalid manifest fields')
    if (type(value['version']) is not int or value['version'] != 2 or value['device'] != 'NPU'
            or value['runtime'] != VERSIONS or value['output_format'] != 'json-schema-v1'):
        raise ValueError('unsupported manifest device or runtime')
    if not isinstance(value['revision'], str) or not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', value['revision']):
        raise ValueError('source revision must be an immutable commit hash')
    for key in ('model_id', 'license', 'export_command'):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 4000:
            raise ValueError('source provenance is required')
    for key, upper in [('max_prompt_tokens', 32768), ('max_new_tokens', 4096)]:
        if type(value[key]) is not int or not 1 <= value[key] <= upper:
            raise ValueError('invalid explicit token limit')
    files = value['files']
    if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys():
        raise ValueError('model file hashes missing')
    for name, digest in files.items():
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError('invalid model file hash')


def create_manifest(model, model_id, revision, license_name, export_command, prompt_tokens, new_tokens):
    value = dict(version=2, device='NPU', runtime=VERSIONS, output_format='json-schema-v1', model_id=model_id,
                 revision=revision, license=license_name, export_command=export_command,
                 max_prompt_tokens=prompt_tokens, max_new_tokens=new_tokens, files=inventory(model))
    validate_manifest(value)
    return value


def verify_model(model, manifest):
    validate_manifest(manifest)
    if inventory(model) != manifest['files']:
        raise ValueError('model files differ from pinned manifest')


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid JSON number')))


def validate_request(value, profile):
    required = {'profile', 'messages', 'schema'}
    if (not isinstance(value, dict) or not required <= value.keys()
            or value.keys() - required - {'max_completion_tokens'}):
        raise ValueError('invalid request')
    if value['profile'] != profile:
        raise RuntimeFailure('PROFILE_CHANGED')
    if not isinstance(value['schema'], dict):
        raise ValueError('invalid schema')
    messages = value['messages']
    if not isinstance(messages, list) or not 1 <= len(messages) <= 128:
        raise ValueError('invalid messages')
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                or message['role'] not in ('system', 'user', 'assistant')
                or not isinstance(message['content'], str) or not message['content'].strip()):
            raise ValueError('invalid message')
    if 'max_completion_tokens' in value:
        limit = value['max_completion_tokens']
        if type(limit) is not int or not 1 <= limit <= 16384:
            raise ValueError('invalid completion token limit')


class OutputRejected(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def validate_content(text):
    try:
        if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_BODY:
            raise ValueError('invalid output text')
        strict_json(text)
    except (ValueError, RecursionError):
        raise OutputRejected('OUTPUT_JSON') from None


def format_prompt(tokenizer, request, limit):
    history = list(request['messages'])
    schema = request['schema']
    kind = 'object' if schema.get('type') == 'object' else 'value'
    instruction = (f'Return one JSON {kind} matching this schema, without markdown:\n'
                   + json.dumps(schema, ensure_ascii=False))
    if history[0]['role'] == 'system':
        history[0] = {'role': 'system', 'content': history[0]['content'] + '\n' + instruction}
    else:
        history.insert(0, {'role': 'system', 'content': instruction})
    prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
    encoded = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
    count = int(encoded.input_ids.shape[-1])
    if count > limit:
        raise ValueError('complete prompt exceeds explicit NPU token limit; nothing was truncated')
    return encoded, count


def worker(connection, model, manifest, cache):
    """Only this child imports OpenVINO; parent can terminate native inference."""
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        stage = 'RUNTIME_VERSION_OR_IMPORT'
        for package, expected in VERSIONS.items():
            if importlib.metadata.version(package) != expected:
                raise ValueError('runtime version differs from pinned profile')
        import openvino as ov
        import openvino_genai as genai
        stage = 'NPU_DEVICE_UNAVAILABLE'
        core = ov.Core()
        if 'NPU' not in core.available_devices:
            raise ValueError('NPU unavailable; automatic fallback is disabled')
        started = time.monotonic()
        stage = 'NPU_COMPILE_OR_MODEL'
        pipeline = genai.LLMPipeline(str(model), 'NPU', {
            'MAX_PROMPT_LEN': manifest['max_prompt_tokens'],
            'MIN_RESPONSE_LEN': manifest['max_new_tokens'], 'CACHE_DIR': str(cache)})
        tokenizer = pipeline.get_tokenizer()
        finish_names = {getattr(genai.GenerationFinishReason, name): name
                        for name in ('NONE', 'STOP', 'LENGTH', 'TOOL_CALL')}
        connection.send({'ready': True, 'load_seconds': time.monotonic()-started})
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            diagnostics = {}
            try:
                stage = 'PROMPT_LIMIT_OR_TEMPLATE'
                encoded, tokens = format_prompt(tokenizer, request, manifest['max_prompt_tokens'])
                started = time.monotonic()
                stage = 'NPU_GENERATION'
                limit = manifest['max_new_tokens']
                limit = min(limit, request.get('max_completion_tokens', limit))
                structured = genai.StructuredOutputConfig(json_schema=json.dumps(request['schema']))
                output = pipeline.generate(encoded, max_new_tokens=limit,
                                           do_sample=False, num_beams=1, num_return_sequences=1,
                                           echo=False, structured_output_config=structured)
                stage = 'OUTPUT_SHAPE'
                sequences, reasons = output.tokens, output.finish_reasons
                diagnostics = {'prompt_tokens': tokens, 'max_new_tokens': limit,
                               'sequence_count': len(sequences), 'finish_reason_count': len(reasons)}
                if len(sequences) == 1:
                    diagnostics['generated_tokens'] = len(sequences[0])
                if len(reasons) == 1:
                    diagnostics['finish_reason'] = finish_names.get(reasons[0], 'UNKNOWN')
                if len(sequences) != 1 or len(reasons) != 1:
                    raise OutputRejected('OUTPUT_SHAPE')
                if len(sequences[0]) >= limit:
                    raise OutputRejected('OUTPUT_TOKEN_LIMIT')
                if reasons[0] != genai.GenerationFinishReason.STOP:
                    raise OutputRejected('OUTPUT_NON_STOP')
                stage = 'OUTPUT_JSON'
                content = tokenizer.decode(sequences[0])
                validate_content(content)
                connection.send({'content': content, 'prompt_tokens': tokens,
                                 'max_new_tokens': limit, 'seconds': time.monotonic()-started})
            except Exception as error:
                if isinstance(error, OutputRejected):
                    stage = error.code
                print(json.dumps({'error': stage, 'device': 'NPU', **diagnostics}), file=sys.stderr, flush=True)
                connection.send({'error': stage})
    except BaseException:
        print(json.dumps({'error': stage, 'device': 'NPU'}), file=sys.stderr, flush=True)
        try:
            connection.send({'error': 'NPU runtime initialization failed; check pinned packages, driver, permissions and model export'})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class Runtime:
    def __init__(self, model, manifest, cache, startup_timeout=600):
        self.snapshot = inventory(model, hashes=False)
        verify_model(model, manifest)
        if inventory(model, hashes=False) != self.snapshot:
            raise ValueError('model changed during hash verification')
        self.model, self.manifest, self.cache = model, manifest, cache
        self.startup_timeout = startup_timeout
        self.profile = profile_id(manifest)
        self.lock = threading.Lock()
        self.recoverable_after_cancel = False
        self.start_failed = False
        self.process = None
        self.connection = None
        self.metadata = {}
        self._start_worker(verify_artifacts=False)

    def _artifacts_match_snapshot(self):
        try:
            return inventory(self.model, hashes=False) == self.snapshot
        except (OSError, ValueError):
            return False

    def _start_worker(self, verify_artifacts=True, call_id=None, recovery=False):
        """Start a clean worker after explicitly verifying the pinned model again."""
        self.recoverable_after_cancel = False
        self.start_failed = True
        try:
            if verify_artifacts:
                try:
                    verify_model(self.model, self.manifest)
                    if not self._artifacts_match_snapshot():
                        raise RuntimeFailure('PROFILE_CHANGED')
                except RuntimeFailure:
                    raise
                except Exception:
                    raise RuntimeFailure('PROFILE_CHANGED') from None
            lifecycle_event('worker_starting', profile=self.profile, stage='NPU_STARTUP', call_id=call_id)
            context = multiprocessing.get_context('spawn')
            connection, child = context.Pipe()
            process = context.Process(target=worker, args=(child, self.model, self.manifest, self.cache), daemon=True)
            try:
                process.start()
            except Exception:
                child.close()
                connection.close()
                raise
            child.close()
            self.connection, self.process = connection, process
            if not connection.poll(self.startup_timeout):
                self._stop_worker(call_id=call_id)
                raise RuntimeError('startup timeout')
            try:
                metadata = connection.recv()
            except EOFError:
                self._stop_worker(call_id=call_id)
                raise RuntimeError('worker exited during startup') from None
            if not metadata.get('ready'):
                self._stop_worker(call_id=call_id)
                raise RuntimeError('worker initialization failed')
            if not self._artifacts_match_snapshot():
                self._stop_worker(call_id=call_id)
                raise RuntimeFailure('PROFILE_CHANGED')
            self.metadata = metadata
            self.recoverable_after_cancel = False
            self.start_failed = False
            lifecycle_event('worker_ready', profile=self.profile, stage='NPU_READY', call_id=call_id)
        except Exception as error:
            if self.process is not None:
                try:
                    self._stop_worker(call_id=call_id)
                except Exception:
                    pass
            self.recoverable_after_cancel = False
            self.start_failed = True
            if isinstance(error, RuntimeFailure):
                raise
            code = 'NPU_RECOVERY_FAILED' if recovery else 'NPU_UNAVAILABLE'
            lifecycle_event('worker_start_failed', profile=self.profile, stage='NPU_STARTUP', call_id=call_id)
            raise RuntimeFailure(code) from None

    def _stop_worker(self, call_id=None):
        process, connection = self.process, self.connection
        had_worker = process is not None
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.is_alive():
                self.start_failed = True
                self.recoverable_after_cancel = False
                raise ValueError('NPU worker could not be stopped safely')
        if connection is not None:
            connection.close()
        self.process = None
        self.connection = None
        if had_worker:
            lifecycle_event('worker_stopped', profile=self.profile, stage='NPU_TEARDOWN', call_id=call_id)

    def close(self):
        self.recoverable_after_cancel = False
        self._stop_worker()

    def _health_locked(self, call_id=None):
        if self.start_failed:
            raise RuntimeFailure('NPU_RECOVERY_FAILED')
        process = self.process
        if not process:
            raise RuntimeFailure('NPU_UNAVAILABLE')
        if not process.is_alive():
            lifecycle_event('worker_exit', profile=self.profile, stage='NPU_HEALTH', call_id=call_id)
            raise RuntimeFailure('NPU_WORKER_EXIT')
        if not self._artifacts_match_snapshot():
            raise RuntimeFailure('PROFILE_CHANGED')
        return {'profile': self.profile, 'device': 'NPU'}

    def status(self, call_id=None):
        """Inspect the pinned worker without recovering it or dispatching a request."""
        if self.recoverable_after_cancel and not self.start_failed:
            if not self._artifacts_match_snapshot():
                raise RuntimeFailure('PROFILE_CHANGED')
            return {'profile': self.profile, 'device': 'NPU', 'loaded': False}
        return dict(self._health_locked(call_id), loaded=True)

    def health(self, call_id=None):
        if self.recoverable_after_cancel:
            with self.lock:
                if self.recoverable_after_cancel:
                    lifecycle_event('worker_recovery_started', profile=self.profile, stage='NPU_STARTUP', call_id=call_id)
                    # Recreate the process only for an observed client disconnect. Timeouts,
                    # crashes, and artifact drift still require an operator restart.
                    self._start_worker(call_id=call_id, recovery=True)
                return self._health_locked(call_id)
        return self._health_locked(call_id)

    def generate(self, request, client, timeout=110, call_id=None):
        validate_request(request, self.profile)
        if not self.lock.acquire(blocking=False):
            raise ValueError('NPU busy; retry explicitly after current request')
        try:
            if self.recoverable_after_cancel:
                lifecycle_event('worker_recovery_started', profile=self.profile, stage='NPU_STARTUP', call_id=call_id)
                self._start_worker(call_id=call_id, recovery=True)
            self._health_locked(call_id)
            self.connection.send(request)
            lifecycle_event('generation_started', profile=self.profile, stage='NPU_GENERATION', call_id=call_id)
            deadline = time.monotonic()+timeout
            while not self.connection.poll(0.1):
                readable, _, _ = select.select([client], [], [], 0)
                try:
                    disconnected = bool(readable) and client.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b''
                except ConnectionResetError:
                    disconnected = True
                if disconnected:
                    self.recoverable_after_cancel = True
                    lifecycle_event('request_cancelled', profile=self.profile, stage='CLIENT_DISCONNECT', call_id=call_id)
                    try:
                        self._stop_worker(call_id=call_id)
                    except Exception:
                        self.recoverable_after_cancel = False
                        raise
                    raise RuntimeFailure('NPU_UNAVAILABLE')
                timed_out = time.monotonic() >= deadline
                worker_exited = not self.process.is_alive()
                if timed_out or worker_exited:
                    reason = 'request_timeout' if timed_out else 'worker_exit'
                    lifecycle_event(reason, profile=self.profile, stage='NPU_GENERATION', call_id=call_id)
                    self._stop_worker(call_id=call_id)
                    raise RuntimeFailure('TIMEOUT' if timed_out else 'NPU_WORKER_EXIT')
            response = self.connection.recv()
            self._health_locked(call_id)
            if 'error' in response:
                lifecycle_event('generation_failed', profile=self.profile,
                                stage=response.get('error', 'NPU_GENERATION'), call_id=call_id)
                code = 'INVALID_OUTPUT' if str(response.get('error', '')).startswith('OUTPUT_') else 'NPU_UNAVAILABLE'
                raise RuntimeFailure(code)
            try:
                validate_content(response.get('content'))
            except OutputRejected:
                raise RuntimeFailure('INVALID_OUTPUT') from None
            print(json.dumps({'device': 'NPU', 'profile': self.profile,
                              'prompt_tokens': response['prompt_tokens'], 'max_new_tokens': response['max_new_tokens'],
                              'seconds': response['seconds']}, ensure_ascii=False), flush=True)
            return dict(self._health_locked(call_id), content=response['content'])
        except (EOFError, BrokenPipeError, ConnectionResetError):
            lifecycle_event('worker_exit', profile=self.profile, stage='NPU_GENERATION', call_id=call_id)
            self._stop_worker(call_id=call_id)
            raise RuntimeFailure('NPU_WORKER_EXIT') from None
        finally:
            self.lock.release()


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def server_bind(self):
        super().server_bind()
        info = os.lstat(self.server_address)
        self.socket_identity = (info.st_dev, info.st_ino)
        os.chmod(self.server_address, 0o660)

    def server_close(self):
        super().server_close()
        try:
            info = os.lstat(self.server_address)
            if (info.st_dev, info.st_ino) == getattr(self, 'socket_identity', None):
                os.unlink(self.server_address)
        except FileNotFoundError:
            pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # Never log prompts or model output.

    def reply(self, status, value):
        data = canonical(value)
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def call_id(self):
        value = self.headers.get('X-NPU-Call-ID', '')
        return value if CALL_ID.fullmatch(value) else None

    @staticmethod
    def failure_status(error):
        return {'TIMEOUT': 504, 'INVALID_OUTPUT': 502, 'PROFILE_CHANGED': 409}.get(error.code, 503)

    def do_GET(self):
        try:
            if self.path == '/status':
                self.reply(200, self.server.runtime.status(self.call_id()))
            elif self.path == '/health':
                self.reply(200, self.server.runtime.health(self.call_id()))
            else:
                self.reply(404, {'error': 'unknown route'})
        except RuntimeFailure as error:
            try:
                self.reply(self.failure_status(error), {'error': 'NPU health check failed', 'code': error.code})
            except OSError:
                pass
        except (ValueError, OSError):
            self.reply(503, {'error': 'NPU health check failed'})

    def do_POST(self):
        try:
            self.connection.settimeout(5)
            if self.path != '/v1/generate' or self.headers.get('Transfer-Encoding'):
                raise ValueError('invalid request')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                raise ValueError('invalid request size')
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError('incomplete request')
            request = strict_json(data.decode('utf-8'))
            self.reply(200, self.server.runtime.generate(request, self.connection, call_id=self.call_id()))
        except RuntimeFailure as error:
            try:
                self.reply(self.failure_status(error), {'error': 'NPU generation failed', 'code': error.code})
            except OSError:
                pass
        except (ValueError, TypeError, KeyError, OSError, EOFError, RecursionError):
            try:
                self.reply(422, {'error': 'NPU generation rejected'})
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('manifest', help='Hash an explicitly exported local model; never downloads')
    prepare.add_argument('--model', type=Path, required=True)
    prepare.add_argument('--model-id', required=True)
    prepare.add_argument('--revision', required=True)
    prepare.add_argument('--license', required=True)
    prepare.add_argument('--export-command', required=True)
    prepare.add_argument('--max-prompt-tokens', type=int, default=2048)
    prepare.add_argument('--max-new-tokens', type=int, default=512)
    prepare.add_argument('--output', type=Path, required=True)
    serve = sub.add_parser('serve')
    serve.add_argument('--model', type=Path, required=True)
    serve.add_argument('--manifest', type=Path, required=True)
    serve.add_argument('--profile', required=True, help='Expected manifest SHA256 from explicit preparation')
    serve.add_argument('--socket', type=Path, required=True)
    serve.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'manifest':
        model = args.model.resolve()
        if args.output.resolve().is_relative_to(model):
            raise ValueError('manifest must be outside model directory')
        manifest = create_manifest(args.model, args.model_id, args.revision, args.license,
                                   args.export_command, args.max_prompt_tokens, args.max_new_tokens)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(manifest)+b'\n')
        print(profile_id(manifest))
        return
    manifest = strict_json(args.manifest.read_bytes())
    if not HEX.fullmatch(args.profile) or profile_id(manifest) != args.profile:
        raise ValueError('manifest does not match explicitly pinned profile')
    if not args.socket.is_absolute() or len(str(args.socket).encode()) > 100:
        raise ValueError('use an absolute Unix socket path under 100 bytes')
    if args.cache.resolve().is_relative_to(args.model.resolve()):
        raise ValueError('compiled cache must be outside model directory')
    args.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if args.socket.exists():
        raise ValueError('socket path exists; stop prior runtime or remove a confirmed stale socket')
    args.cache.mkdir(parents=True, exist_ok=True)
    def stop_server(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop_server)
    runtime = Runtime(args.model, manifest, args.cache)
    print(json.dumps(dict(runtime.metadata, profile=runtime.profile, device='NPU')), flush=True)
    try:
        with UnixHTTPServer(str(args.socket), Handler) as server:
            server.runtime = runtime
            server.serve_forever()
    finally:
        runtime.close()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))
