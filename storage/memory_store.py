import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._state: Dict[str, Any] = {
            "short_term_memory": [],
            "long_term_memory": [],
            "profile": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            self._state.update(loaded)

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "short_term_memory": list(self._state.get("short_term_memory", [])),
                "long_term_memory": list(self._state.get("long_term_memory", [])),
                "profile": dict(self._state.get("profile", {})),
            }

    def append_short_term(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            stm: List[Dict[str, Any]] = self._state.setdefault("short_term_memory", [])
            stm.append(entry)
            self._save()

    def update_long_term(self, entries: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state["long_term_memory"] = entries
            self._save()
