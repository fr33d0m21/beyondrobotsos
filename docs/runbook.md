# BeyondRobotsOS Business Partner Runbook

## Start

```bash
source .venv311/bin/activate
export OPENAI_API_KEY="..."
export CURSOR_API_KEY="..."
export OPENCLAW_BASE_URL="http://127.0.0.1:8788"
export OPENCLAW_API_KEY="..."
export CURSOR_API_BASE_URL="https://cursor.com/api"
export OPENCLAW_ENDPOINT_SPAWN_AGENT="/v1/agents"
export OPENCLAW_ENDPOINT_CONFIGURE_AGENT="/v1/agents/{agent_id}"
export OPENCLAW_ENDPOINT_RUN_TASK="/v1/agents/{agent_id}/tasks"
export OPENCLAW_ENDPOINT_GET_STATUS="/v1/tasks/{task_id}"
export OPENCLAW_ENDPOINT_LIST_AGENTS="/v1/agents"
export OPENCLAW_ENDPOINT_CANCEL_TASK="/v1/tasks/{task_id}:cancel"
export OPENCLAW_ENDPOINT_CAPABILITIES="/v1/capabilities"
export CURSOR_ENDPOINT_CREATE_JOB="/v1/cloud-agents/jobs"
export CURSOR_ENDPOINT_GET_JOB_STATUS="/v1/cloud-agents/jobs/{job_id}"
export CURSOR_ENDPOINT_FOLLOW_UP="/v1/cloud-agents/jobs/{job_id}/messages"
python run_dev.py
```

## Control Plane Endpoints

- `GET /api/state`
- `POST /api/tasks`
- `GET /api/tasks`
- `GET /api/tasks/<task_id>`
- `GET /api/approvals`
- `POST /api/approvals/<task_id>/resolve`
- `GET /api/integrations/health`
- `GET /api/integrations/openclaw/capabilities`

## Integration Wiring Notes

- OpenClaw adapter sends `Authorization: Bearer <OPENCLAW_API_KEY>` when provided.
- OpenClaw spawn/configure calls include idempotency via `Idempotency-Key` header.
- OpenClaw/Cursor clients use HTTP retries for transient failures (429/5xx/timeouts).
- Cursor adapter normalizes response and surfaces `pr_url` when available.

## Safety Policy

- Low risk actions auto-execute by default.
- Medium/high impact actions can require approval.
- Set `BRO_DRY_RUN_DESKTOP_ACTIONS=true` to prevent physical desktop control.

## Recovery

- Audit log is stored at `workspace/runtime/audit.jsonl`.
- Memory snapshot is stored at `workspace/runtime/memory.json`.
- If a run fails, inspect `step_failed` events in the audit log and replay the task intent.
