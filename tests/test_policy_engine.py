import unittest

from config.settings import Settings
from models.tasks import PlanStep, TaskClass
from safety.policy_engine import SafetyPolicyEngine


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            app_name="test",
            openai_api_key="",
            elevenlabs_api_key="",
            host="127.0.0.1",
            port=5000,
            debug=False,
            auto_execute_low_risk=True,
            require_approval_high_impact=True,
            audit_log_path=__import__("pathlib").Path("audit.jsonl"),
            memory_path=__import__("pathlib").Path("memory.json"),
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
        self.engine = SafetyPolicyEngine(self.settings)

    def test_classify_desktop_control(self) -> None:
        task_class = self.engine.classify_intent("Click the save button on my screen")
        self.assertEqual(task_class, TaskClass.desktop_control)

    def test_high_risk_requires_approval(self) -> None:
        step = PlanStep(step_id="s1", tool="openclaw_cancel_task", args={})
        required, details = self.engine.requires_approval(TaskClass.tool_assisted, step)
        self.assertTrue(required)
        self.assertEqual(details["risk"], "high")


if __name__ == "__main__":
    unittest.main()
