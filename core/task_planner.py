from typing import List

from models.tasks import PlanStep, TaskClass


class TaskPlanner:
    def build_plan(self, intent: str, task_class: TaskClass) -> List[PlanStep]:
        lowered = intent.lower()
        steps: List[PlanStep] = []

        if task_class == TaskClass.advice_only:
            steps.append(PlanStep(step_id="advice-1", tool="desktop_get_screenshot", args={}, risk="low"))
            return steps

        if task_class == TaskClass.agent_spawn:
            steps.append(
                PlanStep(
                    step_id="spawn-1",
                    tool="openclaw_spawn_agent",
                    args={"agent_type": "generalPurpose", "config": {"goal": intent}},
                    risk="medium",
                )
            )
            return steps

        if task_class == TaskClass.desktop_control:
            steps.append(PlanStep(step_id="screen-1", tool="desktop_get_screenshot", args={}, risk="low"))
            if "type " in lowered:
                text = intent.split("type ", 1)[1][:300]
                steps.append(PlanStep(step_id="type-1", tool="desktop_type", args={"text": text}, risk="medium"))
            return steps

        if "cursor" in lowered:
            steps.append(
                PlanStep(
                    step_id="cursor-1",
                    tool="cursor_create_cloud_agent_job",
                    args={"repo": "current-repo", "prompt": intent, "branch": "auto/business-partner"},
                    risk="medium",
                )
            )
            return steps

        steps.append(PlanStep(step_id="default-1", tool="desktop_get_screenshot", args={}, risk="low"))
        return steps
