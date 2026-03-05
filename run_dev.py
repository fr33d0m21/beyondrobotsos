import main
from config.settings import load_settings

settings = load_settings()
main.socketio.run(
    main.app,
    host=settings.host,
    port=settings.port,
    debug=settings.debug,
    allow_unsafe_werkzeug=True,
)
