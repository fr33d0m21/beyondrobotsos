from threading import Lock
from typing import Any, Dict, Optional


class ApprovalStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}

    def create(self, task_id: str, reason: str, context: Dict[str, Any]) -> None:
        with self._lock:
            self._pending[task_id] = {
                "task_id": task_id,
                "reason": reason,
                "context": context,
                "approved": None,
            }

    def resolve(self, task_id: str, approved: bool) -> bool:
        with self._lock:
            item = self._pending.get(task_id)
            if not item:
                return False
            item["approved"] = approved
            return True

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._pending.get(task_id)
            if not item:
                return None
            return dict(item)

    def list_pending(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {task_id: dict(value) for task_id, value in self._pending.items()}
