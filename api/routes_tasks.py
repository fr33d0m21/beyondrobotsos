from flask import Blueprint, jsonify, request

from models.tasks import TaskRequest


def build_tasks_blueprint(partner_core) -> Blueprint:
    bp = Blueprint("tasks", __name__, url_prefix="/api/tasks")

    @bp.post("")
    def submit_task():
        payload = request.get_json(force=True, silent=True) or {}
        task_request = TaskRequest(**payload)
        record = partner_core.submit_task(task_request)
        return jsonify(record.model_dump()), 202

    @bp.get("")
    def list_tasks():
        return jsonify(partner_core.list_tasks())

    @bp.get("/<task_id>")
    def get_task(task_id: str):
        record = partner_core.get_task(task_id)
        if not record:
            return jsonify({"error": "task_not_found"}), 404
        return jsonify(record.model_dump())

    return bp
