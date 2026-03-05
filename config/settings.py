import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    openai_api_key: str
    elevenlabs_api_key: str
    host: str
    port: int
    debug: bool
    auto_execute_low_risk: bool
    require_approval_high_impact: bool
    audit_log_path: Path
    memory_path: Path
    openclaw_base_url: str
    openclaw_api_key: str
    openclaw_timeout_seconds: int
    openclaw_endpoint_spawn_agent: str
    openclaw_endpoint_configure_agent: str
    openclaw_endpoint_run_task: str
    openclaw_endpoint_get_status: str
    openclaw_endpoint_list_agents: str
    openclaw_endpoint_cancel_task: str
    openclaw_endpoint_capabilities: str
    cursor_api_base_url: str
    cursor_api_key: str
    cursor_timeout_seconds: int
    cursor_endpoint_create_job: str
    cursor_endpoint_get_job_status: str
    cursor_endpoint_follow_up: str
    dry_run_desktop_actions: bool


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    storage_dir = root / "workspace" / "runtime"
    storage_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=os.getenv("BRO_APP_NAME", "BeyondRobotsOS Business Partner"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
        host=os.getenv("BRO_HOST", "127.0.0.1"),
        port=int(os.getenv("BRO_PORT", "5000")),
        debug=_env_bool("BRO_DEBUG", False),
        auto_execute_low_risk=_env_bool("BRO_AUTO_EXECUTE_LOW_RISK", True),
        require_approval_high_impact=_env_bool("BRO_REQUIRE_APPROVAL_HIGH_IMPACT", True),
        audit_log_path=Path(os.getenv("BRO_AUDIT_LOG_PATH", str(storage_dir / "audit.jsonl"))),
        memory_path=Path(os.getenv("BRO_MEMORY_PATH", str(storage_dir / "memory.json"))),
        openclaw_base_url=os.getenv("OPENCLAW_BASE_URL", "http://127.0.0.1:8788"),
        openclaw_api_key=os.getenv("OPENCLAW_API_KEY", ""),
        openclaw_timeout_seconds=int(os.getenv("OPENCLAW_TIMEOUT_SECONDS", "20")),
        openclaw_endpoint_spawn_agent=os.getenv("OPENCLAW_ENDPOINT_SPAWN_AGENT", "/v1/agents"),
        openclaw_endpoint_configure_agent=os.getenv("OPENCLAW_ENDPOINT_CONFIGURE_AGENT", "/v1/agents/{agent_id}"),
        openclaw_endpoint_run_task=os.getenv("OPENCLAW_ENDPOINT_RUN_TASK", "/v1/agents/{agent_id}/tasks"),
        openclaw_endpoint_get_status=os.getenv("OPENCLAW_ENDPOINT_GET_STATUS", "/v1/tasks/{task_id}"),
        openclaw_endpoint_list_agents=os.getenv("OPENCLAW_ENDPOINT_LIST_AGENTS", "/v1/agents"),
        openclaw_endpoint_cancel_task=os.getenv("OPENCLAW_ENDPOINT_CANCEL_TASK", "/v1/tasks/{task_id}:cancel"),
        openclaw_endpoint_capabilities=os.getenv("OPENCLAW_ENDPOINT_CAPABILITIES", "/v1/capabilities"),
        cursor_api_base_url=os.getenv("CURSOR_API_BASE_URL", "https://cursor.com/api"),
        cursor_api_key=os.getenv("CURSOR_API_KEY", ""),
        cursor_timeout_seconds=int(os.getenv("CURSOR_TIMEOUT_SECONDS", "20")),
        cursor_endpoint_create_job=os.getenv("CURSOR_ENDPOINT_CREATE_JOB", "/v1/cloud-agents/jobs"),
        cursor_endpoint_get_job_status=os.getenv("CURSOR_ENDPOINT_GET_JOB_STATUS", "/v1/cloud-agents/jobs/{job_id}"),
        cursor_endpoint_follow_up=os.getenv("CURSOR_ENDPOINT_FOLLOW_UP", "/v1/cloud-agents/jobs/{job_id}/messages"),
        dry_run_desktop_actions=_env_bool("BRO_DRY_RUN_DESKTOP_ACTIONS", True),
    )
