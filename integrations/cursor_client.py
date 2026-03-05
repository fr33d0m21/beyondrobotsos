from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class CursorClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: int = 20,
        endpoints: Dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.endpoints = endpoints or {
            "create_job": "/v1/cloud-agents/jobs",
            "get_job_status": "/v1/cloud-agents/jobs/{job_id}",
            "follow_up": "/v1/cloud-agents/jobs/{job_id}/messages",
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

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return headers

    def _resolve(self, endpoint_name: str, **path_vars: str) -> str:
        template = self.endpoints[endpoint_name]
        return f"{self.base_url}{template.format(**path_vars)}"

    def _normalize(self, response: requests.Response) -> Dict[str, Any]:
        response.raise_for_status()
        body = response.json() if response.content else {}
        pr_url = None
        if isinstance(body, dict):
            pr_url = body.get("pull_request_url") or body.get("pr_url")
            if not pr_url and isinstance(body.get("links"), dict):
                pr_url = body["links"].get("pull_request")
        return {"ok": True, "status_code": response.status_code, "data": body, "pr_url": pr_url}

    def create_cloud_agent_job(self, repo: str, prompt: str, branch: str) -> Dict[str, Any]:
        response = self.session.post(
            self._resolve("create_job"),
            json={"repo": repo, "prompt": prompt, "branch": branch},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        return self._normalize(response)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        response = self.session.get(
            self._resolve("get_job_status", job_id=job_id),
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        return self._normalize(response)

    def send_follow_up(self, job_id: str, prompt: str) -> Dict[str, Any]:
        response = self.session.post(
            self._resolve("follow_up", job_id=job_id),
            json={"prompt": prompt, "message_type": "follow_up"},
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        return self._normalize(response)
