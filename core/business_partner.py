import threading
import uuid
from typing import Any, Dict, Optional

from integrations.tool_auth import ToolAuthLayer, UserSessionContext
from integrations.tool_registry import ToolRegistry
from models.tasks import TaskRecord, TaskRequest, TaskStatus
from safety.approval_store import ApprovalStore
from safety.policy_engine import SafetyPolicyEngine
from storage.audit_store import AuditStore
from storage.memory_store import MemoryStore
from core.task_planner import TaskPlanner


class BusinessPartnerCore:
    def __init__(
        self,
        planner: TaskPlanner,
        policy: SafetyPolicyEngine,
        approvals: ApprovalStore,
        tools: ToolRegistry,
        memory: MemoryStore,
        audit: AuditStore,
        auth: ToolAuthLayer,
    ) -> None:
        self.planner = planner
        self.policy = policy
        self.approvals = approvals
        self.tools = tools
        self.memory = memory
        self.audit = audit
        self.auth = auth
        self._tasks: Dict[str, TaskRecord] = {}
        self._lock = threading.Lock()

    def submit_task(self, request: TaskRequest) -> TaskRecord:
        task_id = str(uuid.uuid4())
        task_class = self.policy.classify_intent(request.intent)
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.queued,
            task_class=task_class,
            intent=request.intent,
            user_id=request.user_id,
        )
        with self._lock:
            self._tasks[task_id] = record

        self.auth.upsert_session(UserSessionContext(user_id=request.user_id, auth_provider="native"))
        self.audit.append("task_submitted", task_id, {"intent": request.intent, "task_class": task_class.value})
        thread = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
        thread.start()
        return record

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        with self._lock:
            item = self._tasks.get(task_id)
            if not item:
                return None
            return item.model_copy(deep=True)

    def list_tasks(self) -> Dict[str, Any]:
        with self._lock:
            return {task_id: task.model_dump() for task_id, task in self._tasks.items()}

    def resolve_approval(self, task_id: str, approved: bool) -> bool:
        ok = self.approvals.resolve(task_id, approved)
        if ok:
            self.audit.append("approval_resolved", task_id, {"approved": approved})
            record = self.get_task(task_id)
            if record and approved and record.status == TaskStatus.waiting_approval:
                thread = threading.Thread(target=self._run_task, args=(task_id,), daemon=True)
                thread.start()
            elif record and not approved:
                with self._lock:
                    self._tasks[task_id].status = TaskStatus.cancelled
        return ok

    def _set_status(self, task_id: str, status: TaskStatus) -> None:
        with self._lock:
            self._tasks[task_id].status = status

    def _run_task(self, task_id: str) -> None:
        record = self.get_task(task_id)
        if not record:
            return

        if record.status in {TaskStatus.running, TaskStatus.completed, TaskStatus.cancelled}:
            return

        self._set_status(task_id, TaskStatus.planning)
        steps = self.planner.build_plan(record.intent, record.task_class)
        with self._lock:
            self._tasks[task_id].steps = steps

        self.audit.append("task_planned", task_id, {"step_count": len(steps)})
        self._set_status(task_id, TaskStatus.running)

        start_index = 0
        existing = self.get_task(task_id)
        if existing:
            start_index = len(existing.outputs)

        for step in steps[start_index:]:
            requires_approval, detail = self.policy.requires_approval(record.task_class, step)
            if requires_approval:
                self.approvals.create(task_id, detail["reason"], {"step": step.model_dump(), "risk": detail["risk"]})
                self._set_status(task_id, TaskStatus.waiting_approval)
                self.audit.append("approval_required", task_id, {"step": step.model_dump(), "detail": detail})
                return

            try:
                result = self.tools.call(step.tool, step.args)
                with self._lock:
                    self._tasks[task_id].outputs.append({"step_id": step.step_id, "tool": step.tool, "result": result})
                self.memory.append_short_term(
                    {"task_id": task_id, "tool": step.tool, "result_preview": str(result)[:500]}
                )
                self.audit.append("step_completed", task_id, {"step": step.model_dump(), "result": result})
            except Exception as exc:  # pragma: no cover
                with self._lock:
                    self._tasks[task_id].status = TaskStatus.failed
                    self._tasks[task_id].error = str(exc)
                self.audit.append("step_failed", task_id, {"step": step.model_dump(), "error": str(exc)})
                return

        self._set_status(task_id, TaskStatus.completed)
        self.audit.append("task_completed", task_id, {"outputs": len(self.get_task(task_id).outputs if self.get_task(task_id) else [])})
