"""Persistent subprocess client for the isolated Cellpose worker."""

from __future__ import annotations

import atexit
import base64
import json
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
COMMAND_TIMEOUTS = {
    # The first request also waits for isolated-worker third-party imports.
    "probe": 60.0,
    "init": 300.0,
    "set_image": 15.0,
    "predict": 15.0,
    "generate_auto": 300.0,
    "unload": 15.0,
    "shutdown": 5.0,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
CELLPOSE_ENV = REPO_ROOT / "envs" / "cellpose"
CELLPOSE_PYTHON = CELLPOSE_ENV / "bin" / "python"
WORKER_PATH = REPO_ROOT / "auto_annotation" / "cellpose_worker.py"

JsonObject = dict[str, Any]
_QUEUE_EOF = object()
_WRITER_STOP = object()


class CellposeClientError(RuntimeError):
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
        message = f"Cellpose {command} failed ({category}): {detail}"
        if stderr_tail:
            message = f"{message}\nworker stderr tail:\n{stderr_tail}"
        super().__init__(message)
        self.command = command
        self.category = category
        self.stderr_tail = stderr_tail
        self.fatal = fatal


class CellposeProtocolError(CellposeClientError):
    """Raised when the JSONL stream is malformed or desynchronized."""


class CellposeTimeoutError(CellposeClientError):
    """Raised when a worker command exceeds its deadline."""


class CellposeRemoteError(CellposeClientError):
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


def cellpose_interpreter_exists() -> bool:
    """Return whether the isolated Cellpose interpreter exists on disk."""
    return CELLPOSE_PYTHON.is_file()


def _encode_image(image_np: np.ndarray) -> JsonObject:
    image = np.asarray(image_np)
    if image.dtype != np.uint8:
        raise ValueError("Cellpose images must have dtype uint8")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Cellpose images must have shape [H, W, 3]")
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
        raise ValueError("Cellpose Phase 5 does not support tile settings")
    return {
        "image_hash": image_key.image_hash,
        "preprocessing_signature": image_key.preprocessing_signature,
        "tile_settings": None,
    }


class CellposeWorkerClient:
    """Serialize RPCs to one persistent isolated Cellpose process."""

    def __init__(self) -> None:
        if not cellpose_interpreter_exists():
            raise FileNotFoundError(
                f"Cellpose interpreter is missing: {CELLPOSE_PYTHON}"
            )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "VIRTUAL_ENV": str(CELLPOSE_ENV),
                "PYTHONUNBUFFERED": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        self._process = subprocess.Popen(
            [
                str(CELLPOSE_PYTHON),
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
            raise RuntimeError("failed to create Cellpose worker pipes")
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
            name="cellpose-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            name="cellpose-stderr",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._write_stdin,
            args=(self._process.stdin,),
            name="cellpose-stdin",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._writer_thread.start()
        # Normal process cleanup is a fallback; worker PR_SET_PDEATHSIG covers
        # abrupt parent death that bypasses Python cleanup hooks.
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

    def probe(self, model_name: str) -> JsonObject:
        """Return whether the requested model can be loaded in this environment."""
        return self._request("probe", {"model_name": model_name})

    def init(self, model_name: str, device: str = "auto") -> JsonObject:
        """Load a Cellpose model and return worker metadata."""
        return self._request(
            "init",
            {
                "model_name": model_name,
                "device": device,
            },
        )

    def set_image(self, image_np: np.ndarray, image_key: Any) -> JsonObject:
        """Send the defensive set_image command, which Cellpose rejects."""
        started_at = time.monotonic()
        return self._request(
            "set_image",
            {
                "image_key": _encode_image_key(image_key),
                "image": _encode_image(image_np),
            },
            started_at=started_at,
        )

    def predict(
        self,
        *,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
        multimask_output: bool,
    ) -> JsonObject:
        """Send the defensive predict command, which Cellpose rejects."""
        return self._request(
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
        )

    def generate_auto(
        self,
        image_np: np.ndarray,
        image_key: Any,
        params: JsonObject,
    ) -> JsonObject:
        """Run Cellpose automatic segmentation and return polygon records."""
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
    ) -> JsonObject:
        deadline = (started_at or time.monotonic()) + COMMAND_TIMEOUTS[command]
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
            self._raise_fatal(
                command,
                "protocol_mismatch",
                "unexpected protocol version",
            )
        if response.get("id") != request_id:
            self._raise_fatal(command, "id_mismatch", "response ID did not match")
        if response.get("cmd") != command:
            self._raise_fatal(
                command,
                "command_mismatch",
                "response command did not match",
            )
        if response.get("ok") is True:
            if set(response) != {"protocol", "id", "cmd", "ok", "result"}:
                self._raise_fatal(
                    command,
                    "invalid_response",
                    "success envelope is invalid",
                )
            result = response["result"]
            if not isinstance(result, dict):
                self._raise_fatal(
                    command,
                    "invalid_response",
                    "result must be an object",
                )
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
            self._raise_fatal(
                command,
                "invalid_response",
                "error payload types are invalid",
            )
        remote_error = CellposeRemoteError(
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
        raise CellposeTimeoutError(
            command,
            "timeout",
            detail,
            stderr_tail=stderr_tail,
            fatal=True,
        )

    def _raise_fatal(self, command: str, category: str, detail: str) -> None:
        stderr_tail = self.stderr_tail
        self._terminate_process()
        raise CellposeProtocolError(
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
    "CELLPOSE_PYTHON",
    "CellposeClientError",
    "CellposeProtocolError",
    "CellposeRemoteError",
    "CellposeTimeoutError",
    "CellposeWorkerClient",
    "cellpose_interpreter_exists",
]
