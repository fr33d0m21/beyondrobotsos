# AGENTS.md

## Cursor Cloud specific instructions

### Project overview
Beyond Robots OS is a cognitive AI robot operating system — a single Python Flask+SocketIO app (`main.py`) that orchestrates OpenAI LLM modules to simulate human-like cognition. See `README.md` for details. All modules use the `gpt-5.3-codex` model.

### Running the app
- **Entry point:** `python run_dev.py` (wrapper that passes `allow_unsafe_werkzeug=True` to Flask-SocketIO).
  - Do NOT run `python main.py` directly — it will fail with a Werkzeug production-mode error.
- The server starts on port **5000**.
- Requires `OPENAI_API_KEY` env var (the Flask UI loads without it, but the cognitive loop will fail on API calls).
- `ELEVENLABS_API_KEY` is optional (text-to-speech).

### Gotchas
- `playsound==1.3.0` (pinned in `requirements.txt`) fails to build on Python 3.12; install `playsound==1.2.2` instead.
- `openai-whisper==v20231117` needs `--no-build-isolation` and `setuptools<75` (for `pkg_resources`).
- `pipwin` is a Windows-only package and is skipped on Linux.
- `python3.12-dev` and `portaudio19-dev` system packages are required to compile `PyAudio`.
- `ffmpeg` system package is required by `openai-whisper`.
- The `text()` thread in `main.py` reads from stdin; in a non-interactive shell it raises `EOFError` and exits gracefully — this is normal.

### Chat vs Cognitive loop
- The `/send_chat` endpoint returns a hardcoded placeholder (`"AI's response"`) and does **not** call OpenAI. It works without a valid API key or model.
- The cognitive loop (`start_thoght_loop` in `main.py`) runs in a background daemon thread and makes real OpenAI API calls. Its outputs (Thought, Consciousness, Subconsciousness, Answer, Memory) are pushed to the browser via SocketIO. This loop requires a valid `OPENAI_API_KEY` **and** a model the account has access to.

### Linting / Testing / Building
- The project has **no linter**, **no test suite**, and **no build system**. There is no `pyproject.toml`, `setup.py`, `Makefile`, or CI configuration.
- The only validation is running the app and verifying the web UI loads at `http://127.0.0.1:5000/`.
