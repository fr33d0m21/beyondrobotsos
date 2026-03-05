import subprocess
import time
from pathlib import Path
from typing import Any, Dict


class DesktopControlService:
    def __init__(self, artifacts_dir: Path, dry_run: bool = True) -> None:
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

    def get_screenshot(self) -> Dict[str, Any]:
        output = self.artifacts_dir / f"screenshot-{int(time.time() * 1000)}.png"
        subprocess.run(["screencapture", "-x", str(output)], check=True)
        return {"path": str(output)}

    def click(self, x: int, y: int) -> Dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "action": "click", "x": x, "y": y}
        return {"dry_run": False, "action": "click", "note": "Enable real click adapter before production use."}

    def type_text(self, text: str) -> Dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "action": "type", "text": text}
        subprocess.run(["osascript", "-e", f'tell application "System Events" to keystroke "{text}"'], check=True)
        return {"dry_run": False, "action": "type", "text": text}

    def hotkey(self, keys: str) -> Dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "action": "hotkey", "keys": keys}
        return {"dry_run": False, "action": "hotkey", "keys": keys, "note": "Hotkey mapping pending adapter expansion."}

    def scroll(self, amount: int) -> Dict[str, Any]:
        if self.dry_run:
            return {"dry_run": True, "action": "scroll", "amount": amount}
        return {"dry_run": False, "action": "scroll", "amount": amount, "note": "Scroll adapter pending implementation."}

    def wait_for(self, seconds: float) -> Dict[str, Any]:
        time.sleep(seconds)
        return {"action": "wait_for", "seconds": seconds}
