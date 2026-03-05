#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv311/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv311/bin/activate"
fi

export BRO_HOST="${BRO_HOST:-127.0.0.1}"
export BRO_PORT="${BRO_PORT:-5000}"
export BRO_DRY_RUN_DESKTOP_ACTIONS="${BRO_DRY_RUN_DESKTOP_ACTIONS:-true}"

python run_dev.py
