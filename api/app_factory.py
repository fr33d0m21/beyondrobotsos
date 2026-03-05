import threading
import time
from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO
from openai import OpenAI

from config.settings import load_settings
from core.business_partner import BusinessPartnerCore
from core.task_planner import TaskPlanner
from integrations.cursor_client import CursorClient
from integrations.desktop_control import DesktopControlService
from integrations.openclaw_client import OpenClawClient
from integrations.screen_perception import ScreenPerception
from integrations.tool_auth import ToolAuthLayer
from integrations.tool_registry import ToolRegistry
from safety.approval_store import ApprovalStore
from safety.policy_engine import SafetyPolicyEngine
from storage.audit_store import AuditStore
from storage.memory_store import MemoryStore
from api.routes_approvals import build_approvals_blueprint
from api.routes_integrations import build_integrations_blueprint
from api.routes_tasks import build_tasks_blueprint


def create_app() -> tuple[Flask, SocketIO, BusinessPartnerCore]:
    settings = load_settings()

    app = Flask(__name__)
    socketio = SocketIO(app, cors_allowed_origins="*")

    openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else OpenAI()
    _screen = ScreenPerception(openai_client)
    openclaw_client = OpenClawClient(
        base_url=settings.openclaw_base_url,
        api_key=settings.openclaw_api_key,
        timeout_seconds=settings.openclaw_timeout_seconds,
        endpoints={
            "spawn_agent": settings.openclaw_endpoint_spawn_agent,
            "configure_agent": settings.openclaw_endpoint_configure_agent,
            "run_task": settings.openclaw_endpoint_run_task,
            "get_status": settings.openclaw_endpoint_get_status,
            "list_agents": settings.openclaw_endpoint_list_agents,
            "cancel_task": settings.openclaw_endpoint_cancel_task,
            "capabilities": settings.openclaw_endpoint_capabilities,
        },
    )
    cursor_client = CursorClient(
        base_url=settings.cursor_api_base_url,
        api_key=settings.cursor_api_key,
        timeout_seconds=settings.cursor_timeout_seconds,
        endpoints={
            "create_job": settings.cursor_endpoint_create_job,
            "get_job_status": settings.cursor_endpoint_get_job_status,
            "follow_up": settings.cursor_endpoint_follow_up,
        },
    )
    desktop_service = DesktopControlService(
        artifacts_dir=Path(__file__).resolve().parent.parent / "workspace" / "runtime" / "artifacts",
        dry_run=settings.dry_run_desktop_actions,
    )
    tool_registry = ToolRegistry(openclaw=openclaw_client, cursor=cursor_client, desktop=desktop_service)
    planner = TaskPlanner()
    policy = SafetyPolicyEngine(settings)
    approvals = ApprovalStore()
    memory = MemoryStore(settings.memory_path)
    audit = AuditStore(settings.audit_log_path)
    auth = ToolAuthLayer()

    partner = BusinessPartnerCore(
        planner=planner,
        policy=policy,
        approvals=approvals,
        tools=tool_registry,
        memory=memory,
        audit=audit,
        auth=auth,
    )

    app.register_blueprint(build_tasks_blueprint(partner))
    app.register_blueprint(build_approvals_blueprint(partner, approvals))
    app.register_blueprint(build_integrations_blueprint(settings, tool_registry, openclaw_client))

    @app.get("/")
    def index():
        return render_template("indexV.html", app_name=settings.app_name)

    @app.get("/api/state")
    def current_state():
        return {
            "tasks": partner.list_tasks(),
            "approvals": approvals.list_pending(),
            "memory": memory.snapshot(),
            "tools": tool_registry.tool_names(),
        }

    def broadcaster() -> None:
        while True:
            socketio.emit(
                "update",
                {
                    "tasks": partner.list_tasks(),
                    "approvals": approvals.list_pending(),
                    "memory": memory.snapshot(),
                },
            )
            time.sleep(1.0)

    broadcast_thread = threading.Thread(target=broadcaster, daemon=True)
    broadcast_thread.start()

    return app, socketio, partner
