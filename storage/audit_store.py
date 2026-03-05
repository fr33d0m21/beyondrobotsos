import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from models.events import AuditEvent


class AuditStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event_type: str, task_id: str, payload: Dict[str, Any]) -> None:
        event = AuditEvent(event_type=event_type, task_id=task_id, payload=payload)
        line = json.dumps(event.model_dump(), ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
