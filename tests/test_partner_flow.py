import tempfile
import time
import unittest
from pathlib import Path

from config.settings import Settings
from core.business_partner import BusinessPartnerCore
from core.task_planner import TaskPlanner
from integrations.tool_auth import ToolAuthLayer
from models.tasks import TaskRequest, TaskStatus
from safety.approval_store import ApprovalStore
from safety.policy_engine import SafetyPolicyEngine
from storage.audit_store import AuditStore
from storage.memory_store import MemoryStore


class DummyTools:
    def __init__(self) -> None:
        self.calls = []

    def call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return {"ok": True, "tool_name": tool_name}


class PartnerFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        temp_path = Path(self.tmp.name)
        settings = Settings(
            app_name="test",
            openai_api_key="",
            elevenlabs_api_key="",
            host="127.0.0.1",
            port=5000,
            debug=False,
            auto_execute_low_risk=True,
            require_approval_high_impact=True,
            audit_log_path=temp_path / "audit.jsonl",
            memory_path=temp_path / "memory.json",
            openclaw_base_url="http://127.0.0.1:1",
            openclaw_api_key="",
            openclaw_timeout_seconds=10,
            openclaw_endpoint_spawn_agent="/v1/agents",
            openclaw_endpoint_configure_agent="/v1/agents/{agent_id}",
            openclaw_endpoint_run_task="/v1/agents/{agent_id}/tasks",
            openclaw_endpoint_get_status="/v1/tasks/{task_id}",
            openclaw_endpoint_list_agents="/v1/agents",
            openclaw_endpoint_cancel_task="/v1/tasks/{task_id}:cancel",
            openclaw_endpoint_capabilities="/v1/capabilities",
            cursor_api_base_url="http://127.0.0.1:2",
            cursor_api_key="",
            cursor_timeout_seconds=10,
            cursor_endpoint_create_job="/v1/cloud-agents/jobs",
            cursor_endpoint_get_job_status="/v1/cloud-agents/jobs/{job_id}",
            cursor_endpoint_follow_up="/v1/cloud-agents/jobs/{job_id}/messages",
            dry_run_desktop_actions=True,
        )
        self.partner = BusinessPartnerCore(
            planner=TaskPlanner(),
            policy=SafetyPolicyEngine(settings),
            approvals=ApprovalStore(),
            tools=DummyTools(),
            memory=MemoryStore(settings.memory_path),
            audit=AuditStore(settings.audit_log_path),
            auth=ToolAuthLayer(),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_submit_task_completes(self) -> None:
        request = TaskRequest(intent="Take a screenshot and summarize it", user_id="u1")
        record = self.partner.submit_task(request)
        for _ in range(30):
            current = self.partner.get_task(record.task_id)
            if current and current.status in {TaskStatus.completed, TaskStatus.failed, TaskStatus.waiting_approval}:
                break
            time.sleep(0.1)
        current = self.partner.get_task(record.task_id)
        self.assertIsNotNone(current)
        self.assertIn(current.status, {TaskStatus.completed, TaskStatus.waiting_approval})


if __name__ == "__main__":
    unittest.main()
