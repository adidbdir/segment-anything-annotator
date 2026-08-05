"""Persistent subprocess client for the isolated micro-sam worker."""

from __future__ import annotations

import atexit
import base64
import binascii
import json
import math
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

PROTOCOL_VERSION = 1
STDERR_RING_LINES = 200
PROCESS_TERMINATE_TIMEOUT = 2.0
THREAD_JOIN_TIMEOUT = 2.0
INIT_TIMEOUT_CACHED_SECONDS = 120.0
INIT_TIMEOUT_UNCACHED_SECONDS = 1800.0
COMMAND_TIMEOUTS = {
    "probe": 10.0,
    "init": INIT_TIMEOUT_CACHED_SECONDS,
    "set_image": 60.0,
    "predict": 30.0,
    "generate_auto": 180.0,
    "unload": 15.0,
    "shutdown": 5.0,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
MICRO_SAM_ENV = REPO_ROOT / "envs" / "micro_sam"
MICRO_SAM_PYTHON = MICRO_SAM_ENV / "bin" / "python"
WORKER_PATH = REPO_ROOT / "auto_annotation" / "micro_sam_worker.py"

JsonObject = dict[str, Any]
_QUEUE_EOF = object()
_WRITER_STOP = object()


class MicroSamClientError(RuntimeError):
    """Base exception for worker transport and remote-command failures."""

    def __init__(
        self,
        command: str,
        category: str,
        detail: str,
        *,
        stderr_tail: str = "",
        fatal: bool,
    ) -> None:
        message = f"micro-sam {command} failed ({category}): {detail}"
        if stderr_tail:
            message = f"{message}\nworker stderr tail:\n{stderr_tail}"
        super().__init__(message)
        self.command = command
        self.category = category
        self.stderr_tail = stderr_tail
        self.fatal = fatal


class MicroSamProtocolError(MicroSamClientError):
    """Raised when the JSONL stream is malformed or desynchronized."""


class MicroSamTimeoutError(MicroSamClientError):
    """Raised when a worker command exceeds its deadline."""


class MicroSamRemoteError(MicroSamClientError):
    """Raised for a valid ``ok:false`` response from the worker."""

    def __init__(
        self,
        command: str,
        code: str,
        error_type: str,
        detail: str,
        *,
        stderr_tail: str,
        fatal: bool,
    ) -> None:
        super().__init__(
            command,
            code,
            f"{error_type}: {detail}",
            stderr_tail=stderr_tail,
            fatal=fatal,
        )
        self.code = code
        self.error_type = error_type


@dataclass
class _WriteItem:
    line: str
    done: threading.Event
    error: BaseException | None = None


def micro_sam_interpreter_exists() -> bool:
    """Return whether the isolated micro-sam interpreter exists on disk."""
    return MICRO_SAM_PYTHON.is_file()


def _encode_image(image_np: np.ndarray) -> JsonObject:
    image = np.asarray(image_np)
    if image.dtype != np.uint8:
        raise ValueError("micro-sam images must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("micro-sam images must have shape [H, W, 3]")
    contiguous = np.ascontiguousarray(image)
    return {
        "encoding": "raw-base64",
        "dtype": "uint8",
        "shape": list(contiguous.shape),
        "order": "C",
        "nbytes": contiguous.nbytes,
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _encode_image_key(image_key: Any) -> JsonObject:
    if image_key.tile_settings is not None:
        raise ValueError("micro-sam Phase 3 does not support tile settings")
    return {
        "image_hash": image_key.image_hash,
        "preprocessing_signature": image_key.preprocessing_signature,
        "tile_settings": None,
    }


def _decode_base64(value: Any, context: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{context}.data must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{context}.data is invalid base64") from exc


def _decode_packed_bool(payload: Any) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("masks must be an object")
    required = {
        "encoding",
        "dtype",
        "shape",
        "bitorder",
        "count",
        "packed_nbytes",
        "data",
    }
    if set(payload) != required:
        raise ValueError("masks has an invalid schema")
    if (
        payload["encoding"] != "packbits-base64"
        or payload["dtype"] != "bool"
        or payload["bitorder"] != "little"
    ):
        raise ValueError("masks has an unsupported encoding")
    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
    ):
        raise ValueError("masks.shape is invalid")
    count = math.prod(shape)
    if payload["count"] != count:
        raise ValueError("masks.count does not match masks.shape")
    packed = _decode_base64(payload["data"], "masks")
    if payload["packed_nbytes"] != len(packed) or len(packed) != (count + 7) // 8:
        raise ValueError("masks.packed_nbytes is invalid")
    return np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8),
        count=count,
        bitorder="little",
    ).reshape(shape).astype(bool, copy=False)


def _decode_raw_float32(payload: Any, context: str) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    required = {"encoding", "dtype", "shape", "order", "nbytes", "data"}
    if set(payload) != required:
        raise ValueError(f"{context} has an invalid schema")
    if (
        payload["encoding"] != "raw-base64"
        or payload["dtype"] != "<f4"
        or payload["order"] != "C"
    ):
        raise ValueError(f"{context} has an unsupported encoding")
    shape = payload["shape"]
    if not isinstance(shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise ValueError(f"{context}.shape is invalid")
    data = _decode_base64(payload["data"], context)
    expected_nbytes = math.prod(shape) * np.dtype("<f4").itemsize
    if payload["nbytes"] != expected_nbytes or len(data) != expected_nbytes:
        raise ValueError(f"{context}.nbytes is invalid")
    return np.frombuffer(data, dtype="<f4").reshape(shape, order="C")


class MicroSamWorkerClient:
    """Serialize RPCs to one persistent isolated micro-sam process."""

    def __init__(self) -> None:
        if not micro_sam_interpreter_exists():
            raise FileNotFoundError(f"micro-sam interpreter is missing: {MICRO_SAM_PYTHON}")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "VIRTUAL_ENV": str(MICRO_SAM_ENV),
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
                "TQDM_DISABLE": "1",
            }
        )
        self._process = subprocess.Popen(
            [
                str(MICRO_SAM_PYTHON),
                "-I",
                "-u",
                str(WORKER_PATH),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            cwd=REPO_ROOT,
            env=environment,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
        if (
            self._process.stdin is None
            or self._process.stdout is None
            or self._process.stderr is None
        ):
            self._process.kill()
            raise RuntimeError("failed to create micro-sam worker pipes")
        self._responses: queue.Queue[str | object] = queue.Queue()
        self._writes: queue.Queue[_WriteItem | object] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=STDERR_RING_LINES)
        self._stderr_lock = threading.Lock()
        self._rpc_lock = threading.Lock()
        self._request_id = 0
        self._closed = False
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            name="micro-sam-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            name="micro-sam-stderr",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._write_stdin,
            args=(self._process.stdin,),
            name="micro-sam-stdin",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._writer_thread.start()
        # Python cleanup covers normal exit and exception unwinding only.
        # Parent SIGKILL bypasses atexit, __del__, and finally blocks, so the
        # worker-side PR_SET_PDEATHSIG is what closes that orphan-process gap.
        atexit.register(self.terminate)

    def __del__(self) -> None:
        """Best-effort cleanup for clients released before interpreter exit."""
        try:
            if getattr(self, "_process", None) is None or not hasattr(
                self, "_closed"
            ):
                return
            self.terminate()
        except Exception:
            pass

    @property
    def pid(self) -> int:
        """Return the worker PID."""
        return self._process.pid

    @property
    def is_alive(self) -> bool:
        """Return whether the worker subprocess is still running."""
        return not self._closed and self._process.poll() is None

    @property
    def stderr_tail(self) -> str:
        """Return the bounded diagnostic stderr tail."""
        with self._stderr_lock:
            return "".join(self._stderr_lines).rstrip()

    def probe(self, model_type: str) -> JsonObject:
        """Return whether all required checkpoints already exist locally."""
        return self._request("probe", {"model_type": model_type})

    def init(
        self,
        model_type: str,
        device: str = "auto",
        *,
        cached: bool = True,
    ) -> JsonObject:
        """Load a micro-sam model and return its worker metadata."""
        timeout = (
            INIT_TIMEOUT_CACHED_SECONDS
            if cached
            else INIT_TIMEOUT_UNCACHED_SECONDS
        )
        return self._request(
            "init",
            {"model_type": model_type, "device": device},
            timeout_override=timeout,
        )

    def set_image(self, image_np: np.ndarray, image_key: Any) -> JsonObject:
        """Set or restore the active image embedding."""
        started_at = time.monotonic()
        return self._request(
            "set_image",
            {"image_key": _encode_image_key(image_key), "image": _encode_image(image_np)},
            started_at=started_at,
        )

    def predict(
        self,
        *,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
        multimask_output: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run prompt prediction and decode NumPy result arrays."""
        started_at = time.monotonic()
        result = self._request(
            "predict",
            {
                "point_coords": (
                    None if point_coords is None else np.asarray(point_coords).tolist()
                ),
                "point_labels": (
                    None if point_labels is None else np.asarray(point_labels).tolist()
                ),
                "box": None if box is None else np.asarray(box).tolist(),
                "multimask_output": multimask_output,
            },
            started_at=started_at,
        )
        try:
            masks = _decode_packed_bool(result["masks"])
            score_payload = result["scores"]
            if not isinstance(score_payload, dict) or set(score_payload) != {
                "dtype",
                "shape",
                "data",
            }:
                raise ValueError("scores has an invalid schema")
            if score_payload["dtype"] != "<f4":
                raise ValueError("scores dtype is invalid")
            scores = np.asarray(score_payload["data"], dtype="<f4")
            if list(scores.shape) != score_payload["shape"]:
                raise ValueError("scores shape is invalid")
            low_res_logits = _decode_raw_float32(
                result["low_res_logits"], "low_res_logits"
            )
        except (KeyError, TypeError, ValueError) as exc:
            self._raise_fatal("predict", "invalid_result", str(exc))
        return masks, scores, low_res_logits

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: Any,
        params: JsonObject,
    ) -> JsonObject:
        """Run automatic segmentation and return polygon-only records."""
        started_at = time.monotonic()
        return self._request(
            "generate_auto",
            {
                "image_key": _encode_image_key(image_key),
                "image": _encode_image(image_np),
                "params": params,
            },
            started_at=started_at,
        )

    def unload(self) -> JsonObject:
        """Unload model state while keeping the worker alive."""
        return self._request("unload", {})

    def shutdown(self) -> JsonObject:
        """Ask the worker to unload and exit cleanly."""
        result = self._request("shutdown", {})
        try:
            self._process.wait(timeout=COMMAND_TIMEOUTS["shutdown"])
        except subprocess.TimeoutExpired:
            self._raise_fatal("shutdown", "exit_timeout", "worker did not exit")
        self._finish_transport()
        return result

    def terminate(self) -> None:
        """Unconditionally terminate the worker and close transport threads."""
        self._terminate_process()

    def _request(
        self,
        command: str,
        payload: JsonObject,
        *,
        started_at: float | None = None,
        timeout_override: float | None = None,
    ) -> JsonObject:
        timeout = (
            timeout_override
            if timeout_override is not None
            else COMMAND_TIMEOUTS[command]
        )
        deadline = (started_at or time.monotonic()) + timeout
        with self._rpc_lock:
            if not self.is_alive:
                self._raise_fatal(command, "worker_dead", "worker is not running")
            try:
                unexpected = self._responses.get_nowait()
            except queue.Empty:
                unexpected = None
            if unexpected is not None:
                self._raise_fatal(
                    command,
                    "unexpected_response",
                    "response queue was not empty before request",
                )

            self._request_id += 1
            request_id = self._request_id
            request = {
                "protocol": PROTOCOL_VERSION,
                "id": request_id,
                "cmd": command,
                **payload,
            }
            try:
                line = json.dumps(
                    request,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {command} request: {exc}") from exc

            write_item = _WriteItem(line=line, done=threading.Event())
            self._writes.put(write_item)
            if not write_item.done.wait(timeout=self._remaining(deadline)):
                self._raise_timeout(command, "request write timed out")
            if write_item.error is not None:
                self._raise_fatal(command, "write_failed", str(write_item.error))
            try:
                response_line = self._responses.get(timeout=self._remaining(deadline))
            except queue.Empty:
                self._raise_timeout(command, "response timed out")
            if response_line is _QUEUE_EOF:
                self._raise_fatal(command, "worker_eof", "worker closed stdout")
            if not isinstance(response_line, str):
                self._raise_fatal(command, "invalid_response", "response was not text")
            return self._parse_response(response_line, request_id, command)

    def _parse_response(
        self,
        line: str,
        request_id: int,
        command: str,
    ) -> JsonObject:
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            self._raise_fatal(command, "malformed_json", str(exc))
        if not isinstance(response, dict):
            self._raise_fatal(command, "invalid_response", "response must be an object")
        if response.get("protocol") != PROTOCOL_VERSION:
            self._raise_fatal(command, "protocol_mismatch", "unexpected protocol version")
        if response.get("id") != request_id:
            self._raise_fatal(command, "id_mismatch", "response ID did not match")
        if response.get("cmd") != command:
            self._raise_fatal(command, "command_mismatch", "response command did not match")
        if response.get("ok") is True:
            if set(response) != {"protocol", "id", "cmd", "ok", "result"}:
                self._raise_fatal(command, "invalid_response", "success envelope is invalid")
            result = response["result"]
            if not isinstance(result, dict):
                self._raise_fatal(command, "invalid_response", "result must be an object")
            return result
        if response.get("ok") is not False or set(response) != {
            "protocol",
            "id",
            "cmd",
            "ok",
            "error",
        }:
            self._raise_fatal(command, "invalid_response", "error envelope is invalid")
        error = response["error"]
        if not isinstance(error, dict) or set(error) != {
            "code",
            "type",
            "message",
            "fatal",
        }:
            self._raise_fatal(command, "invalid_response", "error payload is invalid")
        fatal = error["fatal"]
        if not isinstance(fatal, bool) or not all(
            isinstance(error[field], str) for field in ("code", "type", "message")
        ):
            self._raise_fatal(command, "invalid_response", "error payload types are invalid")
        remote_error = MicroSamRemoteError(
            command,
            error["code"],
            error["type"],
            error["message"],
            stderr_tail=self.stderr_tail,
            fatal=fatal,
        )
        if fatal:
            self._terminate_process()
        raise remote_error

    @staticmethod
    def _remaining(deadline: float) -> float:
        return max(0.0, deadline - time.monotonic())

    def _raise_timeout(self, command: str, detail: str) -> None:
        stderr_tail = self.stderr_tail
        self._terminate_process()
        raise MicroSamTimeoutError(
            command,
            "timeout",
            detail,
            stderr_tail=stderr_tail,
            fatal=True,
        )

    def _raise_fatal(self, command: str, category: str, detail: str) -> None:
        stderr_tail = self.stderr_tail
        self._terminate_process()
        raise MicroSamProtocolError(
            command,
            category,
            detail,
            stderr_tail=stderr_tail,
            fatal=True,
        )

    def _read_stdout(self, stream: TextIO) -> None:
        try:
            while True:
                line = stream.readline()
                if line == "":
                    break
                self._responses.put(line)
        finally:
            self._responses.put(_QUEUE_EOF)

    def _read_stderr(self, stream: TextIO) -> None:
        while True:
            line = stream.readline()
            if line == "":
                return
            with self._stderr_lock:
                self._stderr_lines.append(line)

    def _write_stdin(self, stream: TextIO) -> None:
        while True:
            item = self._writes.get()
            if item is _WRITER_STOP:
                return
            if not isinstance(item, _WriteItem):
                continue
            try:
                stream.write(f"{item.line}\n")
                stream.flush()
            except BaseException as exc:  # noqa: BLE001 - transport boundary
                item.error = exc
            finally:
                item.done.set()

    def _terminate_process(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_stdin()
        if self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait(timeout=PROCESS_TERMINATE_TIMEOUT)
        self._finish_transport()

    def _finish_transport(self) -> None:
        self._closed = True
        self._close_stdin()
        self._writes.put(_WRITER_STOP)
        for thread in (
            self._writer_thread,
            self._stdout_thread,
            self._stderr_thread,
        ):
            if thread is not threading.current_thread():
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def _close_stdin(self) -> None:
        stream = self._process.stdin
        if stream is None or stream.closed:
            return
        try:
            stream.close()
        except OSError:
            pass


__all__ = [
    "MICRO_SAM_PYTHON",
    "MicroSamClientError",
    "MicroSamProtocolError",
    "MicroSamRemoteError",
    "MicroSamTimeoutError",
    "MicroSamWorkerClient",
    "micro_sam_interpreter_exists",
]
