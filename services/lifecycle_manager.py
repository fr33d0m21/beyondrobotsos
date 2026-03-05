import signal
import threading
from typing import Callable, List


class LifecycleManager:
    def __init__(self) -> None:
        self._callbacks: List[Callable[[], None]] = []
        self._lock = threading.Lock()
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def register_shutdown(self, callback: Callable[[], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def _handle_shutdown(self, _sig, _frame) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass
