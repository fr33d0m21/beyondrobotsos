from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskClass(str, Enum):
    advice_only = "advice_only"
    tool_assisted = "tool_assisted"
    agent_spawn = "agent_spawn"
    desktop_control = "desktop_control"
    high_impact = "high_impact"


class TaskStatus(str, Enum):
    queued = "queued"
    planning = "planning"
    waiting_approval = "waiting_approval"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskRequest(BaseModel):
    intent: str = Field(min_length=1, max_length=5000)
    user_id: str = Field(default="default-user", min_length=1, max_length=200)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    step_id: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    risk: str = "low"


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.queued
    task_class: TaskClass = TaskClass.tool_assisted
    intent: str
    user_id: str
    steps: List[PlanStep] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
