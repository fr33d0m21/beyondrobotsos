from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    event_type: str
    task_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
