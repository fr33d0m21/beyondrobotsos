from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class UserSessionContext:
    user_id: str
    account_alias: str = "default"
    auth_provider: str = "native"
    connection_meta: Dict[str, Any] | None = None


class ToolAuthLayer:
    def __init__(self) -> None:
        self._sessions: Dict[str, UserSessionContext] = {}

    def upsert_session(self, context: UserSessionContext) -> None:
        self._sessions[context.user_id] = context

    def get_session(self, user_id: str) -> UserSessionContext:
        return self._sessions.get(user_id, UserSessionContext(user_id=user_id))
