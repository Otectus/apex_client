import atexit
import queue
import threading
import time
from multiprocessing import Queue
from typing import Callable, Dict, Any, Optional

from .protocol import (
    ENGINE_MODES,
    REQUEST_FORGET,
    REQUEST_HEALTH,
    REQUEST_INGEST,
    REQUEST_SEARCH,
    REQUEST_SHUTDOWN,
    EngineRequest,
)
from .worker import MemoryWorker


class MemoryEngineClient:
    def __init__(
        self,
        config_builder: Callable[[], Dict[str, Any]],
        group_id_provider: Callable[[], str],
        logger: Optional[Callable[[str], None]] = None,
        error_logger: Optional[Callable[[str], None]] = None,
    ):
        self.config_builder = config_builder
        self.group_id_provider = group_id_provider
        self.logger = logger or (lambda msg: None)
        self.error_logger = error_logger or (lambda msg: None)
        self.request_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.worker: Optional[MemoryWorker] = None
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._pending_condition = threading.Condition(self._lock)
        self._external_polling = False
        atexit.register(self.shutdown)

    def _log(self, msg: str):
        try:
            self.logger(msg)
        except Exception:
            pass

    def _error(self, msg: str):
        try:
            self.error_logger(msg)
        except Exception:
            pass

    def _start_worker(self) -> bool:
        if self.worker and self.worker.is_alive():
            return True

        self._log("Starting persistent Graphiti worker")
        config = self.config_builder()
        group_id = self.group_id_provider()
        self.worker = MemoryWorker(config=config, group_id=group_id, request_queue=self.request_queue, response_queue=self.response_queue)
        self.worker.start()
        return self.worker.is_alive()

    def restart(self) -> bool:
        self.shutdown()
        return self._start_worker()

    def start(self) -> bool:
        return self._start_worker()

    def shutdown(self):
        if not self.worker:
            return
        try:
            req = EngineRequest(operation=REQUEST_SHUTDOWN)
            self.request_queue.put(req.to_dict())
            self.worker.join(timeout=5)
        except Exception:
            pass
        finally:
            if self.worker.is_alive():
                try:
                    self.worker.terminate()
                except Exception:
                    pass
            self.worker = None

    def is_alive(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def enable_external_polling(self):
        """Allow an external thread to drain the response queue."""
        self._external_polling = True

    def _pop_pending(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._pending_condition:
            return self._pending.pop(request_id, None)

    def _store_pending(self, request_id: str, payload: Dict[str, Any]):
        if not request_id:
            return
        with self._pending_condition:
            self._pending[request_id] = payload
            self._pending_condition.notify_all()

    def poll_response(self, timeout: float = 0.5) -> Optional[Dict[str, Any]]:
        """Retrieve a raw response from the worker queue (used by plugin poller)."""
        if not self._external_polling:
            return None
        if not self.worker or not self.worker.is_alive():
            return None
        try:
            resp = self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        except Exception:
            return None
        rid = resp.get("request_id")
        self._store_pending(rid, resp)
        return resp

    def submit_async(self, operation: str, payload: Dict[str, Any]) -> Optional[str]:
        """Queue a request without waiting for a response."""
        if not self._start_worker():
            self._error("Persistent worker is not available")
            return None
        request = EngineRequest(operation=operation, payload=payload)
        try:
            self.request_queue.put(request.to_dict())
            return request.request_id
        except Exception as e:
            self._error(f"Failed to enqueue request: {e}")
            return None

    def _wait_for_response_polling(self, request_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        end_time = time.time() + timeout
        with self._pending_condition:
            resp = self._pending.pop(request_id, None)
            if resp:
                return resp
            while time.time() < end_time:
                remaining = end_time - time.time()
                if remaining <= 0:
                    break
                self._pending_condition.wait(timeout=remaining)
                resp = self._pending.pop(request_id, None)
                if resp:
                    return resp
        return None

    def _wait_for_response(self, request_id: str, timeout: float) -> Optional[Dict[str, Any]]:
        if self._external_polling:
            resp = self._wait_for_response_polling(request_id, timeout)
            if resp:
                return resp
            return None

        pending = self._pop_pending(request_id)
        if pending:
            return pending

        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                resp = self.response_queue.get(timeout=max(0.1, end_time - time.time()))
                rid = resp.get("request_id")
                if rid == request_id:
                    return resp
                self._store_pending(rid, resp)
            except queue.Empty:
                break
            except Exception:
                break
        return None

    def _request(self, operation: str, payload: Dict[str, Any], timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        if not self._start_worker():
            self._error("Persistent worker is not available")
            return None

        request = EngineRequest(operation=operation, payload=payload)
        try:
            self.request_queue.put(request.to_dict())
        except Exception as e:
            self._error(f"Failed to enqueue request: {e}")
            return None

        resp = self._wait_for_response(request.request_id, timeout)
        if not resp:
            self._error("Timed out waiting for engine response")
        return resp

    def health(self) -> Optional[Dict[str, Any]]:
        return self._request(REQUEST_HEALTH, {}, timeout=5.0)

    def ingest(self, name: str, content: str, mode: str) -> Optional[Dict[str, Any]]:
        return self._request(REQUEST_INGEST, {"name": name, "content": content, "mode": mode})

    def search(self, query: str, limit: int) -> Optional[Dict[str, Any]]:
        return self._request(REQUEST_SEARCH, {"query": query, "limit": limit})

    def forget(self, query: str) -> Optional[Dict[str, Any]]:
        return self._request(REQUEST_FORGET, {"query": query})


__all__ = [
    "MemoryEngineClient",
    "ENGINE_MODES",
]
