from flask import Flask
from flask_login import LoginManager
# from config import Config

# login_manager = LoginManager()
# login_manager.login_view = 'auth.login'

def create_app():
    app = Flask(__name__)
    # app.config.from_object(Config)

    from app.routes import main
    # app.register_blueprint(auth)
    app.register_blueprint(main)
    
    return app