import uuid
from typing import Any, Callable, Dict

from integrations.cursor_client import CursorClient
from integrations.desktop_control import DesktopControlService
from integrations.openclaw_client import OpenClawClient


ToolFunction = Callable[..., Dict[str, Any]]


class ToolRegistry:
    def __init__(
        self,
        openclaw: OpenClawClient,
        cursor: CursorClient,
        desktop: DesktopControlService,
    ) -> None:
        self.openclaw = openclaw
        self.cursor = cursor
        self.desktop = desktop
        self._tools: Dict[str, ToolFunction] = {
            "openclaw_spawn_agent": self._openclaw_spawn_agent,
            "openclaw_configure_agent": self._openclaw_configure_agent,
            "openclaw_run_task": self._openclaw_run_task,
            "openclaw_get_status": self._openclaw_get_status,
            "openclaw_list_agents": self._openclaw_list_agents,
            "openclaw_cancel_task": self._openclaw_cancel_task,
            "desktop_get_screenshot": self._desktop_get_screenshot,
            "desktop_click": self._desktop_click,
            "desktop_type": self._desktop_type,
            "desktop_hotkey": self._desktop_hotkey,
            "desktop_scroll": self._desktop_scroll,
            "desktop_wait_for": self._desktop_wait_for,
            "cursor_create_cloud_agent_job": self._cursor_create_job,
            "cursor_get_job_status": self._cursor_get_job_status,
            "cursor_send_follow_up": self._cursor_send_follow_up,
        }

    def tool_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._tools.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return fn(**args)

    def _openclaw_spawn_agent(self, agent_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.openclaw.spawn_agent(agent_type, config, idempotency_key=str(uuid.uuid4()))

    def _openclaw_configure_agent(self, agent_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        return self.openclaw.configure_agent(agent_id, config, idempotency_key=str(uuid.uuid4()))

    def _openclaw_run_task(self, agent_id: str, task: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.openclaw.run_task(agent_id, task, metadata or {})

    def _openclaw_get_status(self, task_id: str) -> Dict[str, Any]:
        return self.openclaw.get_status(task_id)

    def _openclaw_list_agents(self) -> Dict[str, Any]:
        return self.openclaw.list_agents()

    def _openclaw_cancel_task(self, task_id: str) -> Dict[str, Any]:
        return self.openclaw.cancel_task(task_id)

    def _desktop_get_screenshot(self) -> Dict[str, Any]:
        return self.desktop.get_screenshot()

    def _desktop_click(self, x: int, y: int) -> Dict[str, Any]:
        return self.desktop.click(x=x, y=y)

    def _desktop_type(self, text: str) -> Dict[str, Any]:
        return self.desktop.type_text(text=text)

    def _desktop_hotkey(self, keys: str) -> Dict[str, Any]:
        return self.desktop.hotkey(keys=keys)

    def _desktop_scroll(self, amount: int) -> Dict[str, Any]:
        return self.desktop.scroll(amount=amount)

    def _desktop_wait_for(self, seconds: float) -> Dict[str, Any]:
        return self.desktop.wait_for(seconds=seconds)

    def _cursor_create_job(self, repo: str, prompt: str, branch: str) -> Dict[str, Any]:
        return self.cursor.create_cloud_agent_job(repo=repo, prompt=prompt, branch=branch)

    def _cursor_get_job_status(self, job_id: str) -> Dict[str, Any]:
        return self.cursor.get_job_status(job_id=job_id)

    def _cursor_send_follow_up(self, job_id: str, prompt: str) -> Dict[str, Any]:
        return self.cursor.send_follow_up(job_id=job_id, prompt=prompt)
