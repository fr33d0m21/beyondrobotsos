from flask import Blueprint, jsonify


def build_integrations_blueprint(settings, tools, openclaw) -> Blueprint:
    bp = Blueprint("integrations", __name__, url_prefix="/api/integrations")

    @bp.get("/health")
    def health():
        return jsonify(
            {
                "app": settings.app_name,
                "openclaw_base_url": settings.openclaw_base_url,
                "openclaw_endpoints": {
                    "spawn_agent": settings.openclaw_endpoint_spawn_agent,
                    "configure_agent": settings.openclaw_endpoint_configure_agent,
                    "run_task": settings.openclaw_endpoint_run_task,
                    "get_status": settings.openclaw_endpoint_get_status,
                    "list_agents": settings.openclaw_endpoint_list_agents,
                    "cancel_task": settings.openclaw_endpoint_cancel_task,
                    "capabilities": settings.openclaw_endpoint_capabilities,
                },
                "cursor_api_base_url": settings.cursor_api_base_url,
                "cursor_endpoints": {
                    "create_job": settings.cursor_endpoint_create_job,
                    "get_job_status": settings.cursor_endpoint_get_job_status,
                    "follow_up": settings.cursor_endpoint_follow_up,
                },
                "tools": tools.tool_names(),
            }
        )

    @bp.get("/openclaw/capabilities")
    def openclaw_capabilities():
        try:
            return jsonify(openclaw.capabilities())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 502

    return bp
