from typing import Any, Dict, Tuple

from config.settings import Settings
from models.tasks import PlanStep, TaskClass


class SafetyPolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def classify_intent(self, intent: str) -> TaskClass:
        lowered = intent.lower()
        if any(keyword in lowered for keyword in ("spawn", "configure agent", "openclaw")):
            return TaskClass.agent_spawn
        if any(keyword in lowered for keyword in ("click", "type", "screen", "desktop", "keyboard", "mouse")):
            return TaskClass.desktop_control
        if any(keyword in lowered for keyword in ("delete", "shutdown", "payment", "purchase", "transfer")):
            return TaskClass.high_impact
        if any(keyword in lowered for keyword in ("plan", "research", "summarize", "analyze")):
            return TaskClass.advice_only
        return TaskClass.tool_assisted

    def step_risk(self, step: PlanStep) -> str:
        tool = step.tool.lower()
        if "cancel" in tool or "delete" in tool:
            return "high"
        if "desktop_" in tool and ("type" in tool or "hotkey" in tool):
            return "medium"
        if "spawn" in tool or "configure" in tool:
            return "medium"
        return "low"

    def requires_approval(self, task_class: TaskClass, step: PlanStep) -> Tuple[bool, Dict[str, Any]]:
        risk = self.step_risk(step)
        if task_class == TaskClass.high_impact:
            return True, {"risk": "high", "reason": "high impact task class"}
        if risk == "high" and self.settings.require_approval_high_impact:
            return True, {"risk": "high", "reason": "high risk tool action"}
        if risk == "medium" and not self.settings.auto_execute_low_risk:
            return True, {"risk": "medium", "reason": "medium risk requires review in current policy"}
        return False, {"risk": risk, "reason": "auto-approved"}
