from api.app_factory import create_app


app, socketio, partner_core = create_app()


if __name__ == "__main__":
    socketio.run(app)