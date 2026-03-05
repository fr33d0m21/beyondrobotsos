# Beyond Robots OS

## Introduction
Beyond Robots OS is now evolving into a business-partner control plane for local agent operations. It combines:

- an intent-driven orchestration core,
- policy-gated tool execution,
- OpenClaw and Cursor adapters,
- desktop screenshot and action tooling,
- audit/memory persistence for replay and trust.

## Features
- **Business Partner Core**: Intake, planning, routing, execution, and reflection lifecycle.
- **Semi-Autonomous Policy**: Auto-exec low risk actions, approval for high-impact actions.
- **OpenClaw Tooling**: Spawn/configure/run/status/list/cancel primitives.
- **Cursor Tooling**: Cloud-agent job create/status/follow-up primitives.
- **Desktop Control Pipeline**: Screenshot, click, type, hotkey, scroll, wait tools.
- **Operator Console**: Task queue, approvals, memory timeline, and live updates.
- **Audit + Memory Stores**: Persistent JSON/JSONL run traces.

## How it Works
The control plane is modular:

- `core/` for planning and orchestration.
- `safety/` for risk and approvals.
- `integrations/` for OpenClaw, Cursor, and desktop tools.
- `api/` for HTTP endpoints.
- `storage/` for memory + audit persistence.

## API Endpoints
- `GET /api/state`
- `POST /api/tasks`
- `GET /api/tasks`
- `GET /api/tasks/<task_id>`
- `GET /api/approvals`
- `POST /api/approvals/<task_id>/resolve`
- `GET /api/integrations/health`
- `GET /api/integrations/openclaw/capabilities`

## Installation
Clone and install:

```bash
git clone https://github.com/fr33d0m21/beyondrobotsos/
cd beyondrobotsos
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -r requirements.txt

```

## Set API Keys

```bash
export OPENAI_API_KEY="..."
export ELEVENLABS_API_KEY="..."
export OPENCLAW_BASE_URL="http://127.0.0.1:8788"
export CURSOR_API_KEY="..."
export BRO_DRY_RUN_DESKTOP_ACTIONS=true
```

## Usage
Start the system using:
```bash
python run_dev.py
```
Open `http://127.0.0.1:5000/` and use the operator console.

## Tests
Run:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Runbook
See `docs/runbook.md` for operational and recovery procedures.

## License
Beyond Robots OS is licensed under the [GPL v3 License](LICENSE.md).

## Contributing
Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) for details on how to contribute to the project.

## Contact
For support or inquiries, please contact [instantmarketingnerds@gmail.com].
