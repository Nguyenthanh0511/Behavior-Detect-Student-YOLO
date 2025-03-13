import os
from app.__init___ import create_app
# from app.routes import init_db
app = create_app()

if __name__ == '__main__':
    # init_db()
    app.run(host='0.0.0.0', port=5000, threaded=True)