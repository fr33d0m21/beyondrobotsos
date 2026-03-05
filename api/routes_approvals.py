from flask import Blueprint, jsonify, request


def build_approvals_blueprint(partner_core, approvals) -> Blueprint:
    bp = Blueprint("approvals", __name__, url_prefix="/api/approvals")

    @bp.get("")
    def list_approvals():
        return jsonify(approvals.list_pending())

    @bp.post("/<task_id>/resolve")
    def resolve_approval(task_id: str):
        payload = request.get_json(force=True, silent=True) or {}
        approved = bool(payload.get("approved", False))
        ok = partner_core.resolve_approval(task_id, approved)
        if not ok:
            return jsonify({"error": "approval_not_found"}), 404
        return jsonify({"ok": True, "task_id": task_id, "approved": approved})

    return bp
