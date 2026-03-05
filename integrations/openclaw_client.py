from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class OpenClawClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_seconds: int = 20,
        endpoints: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.endpoints = endpoints or {
            "spawn_agent": "/v1/agents",
            "configure_agent": "/v1/agents/{agent_id}",
            "run_task": "/v1/agents/{agent_id}/tasks",
            "get_status": "/v1/tasks/{task_id}",
            "list_agents": "/v1/agents",
            "cancel_task": "/v1/tasks/{task_id}:cancel",
            "capabilities": "/v1/capabilities",
        }
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[408, 409, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _resolve(self, name: str, **path_vars: str) -> str:
        template = self.endpoints[name]
        endpoint = template.format(**path_vars)
        return f"{self.base_url}{endpoint}"

    def _post(self, endpoint: str, payload: Dict[str, Any], idempotency_key: Optional[str] = None) -> Dict[str, Any]:
        response = self.session.post(
            endpoint,
            json=payload,
            headers=self._headers(idempotency_key=idempotency_key),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json() if response.content else {}
        return {"ok": True, "status_code": response.status_code, "data": body}

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self.session.get(
            endpoint,
            params=params or {},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json() if response.content else {}
        return {"ok": True, "status_code": response.status_code, "data": body}

    def spawn_agent(self, agent_type: str, config: Dict[str, Any], idempotency_key: str) -> Dict[str, Any]:
        endpoint = self._resolve("spawn_agent")
        payload = {"agent_type": agent_type, "config": config}
        return self._post(endpoint, payload, idempotency_key=idempotency_key)

    def configure_agent(self, agent_id: str, config: Dict[str, Any], idempotency_key: str) -> Dict[str, Any]:
        endpoint = self._resolve("configure_agent", agent_id=agent_id)
        payload = {"config": config}
        return self._post(endpoint, payload, idempotency_key=idempotency_key)

    def run_task(self, agent_id: str, task: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = self._resolve("run_task", agent_id=agent_id)
        payload = {"task": task, "metadata": metadata}
        return self._post(endpoint, payload)

    def get_status(self, task_id: str) -> Dict[str, Any]:
        endpoint = self._resolve("get_status", task_id=task_id)
        return self._get(endpoint)

    def list_agents(self) -> Dict[str, Any]:
        endpoint = self._resolve("list_agents")
        return self._get(endpoint)

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        endpoint = self._resolve("cancel_task", task_id=task_id)
        return self._post(endpoint, {})

    def capabilities(self) -> Dict[str, Any]:
        endpoint = self._resolve("capabilities")
        return self._get(endpoint)
