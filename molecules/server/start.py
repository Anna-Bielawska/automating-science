if __name__ == "__main__":
    import sklearn
    import sys
    import re
    from sys import platform
    assert sklearn.__version__ == '1.1.0', ('Exact sklearn version is needed because PyTDC uses pickled random forest'
                                        'that requires exact version match.')


    project_dir: str = sys.path[0]

    if "linux" in platform or platform == "darwin":
        project_dir = re.sub(r'\/server', '', sys.path[0])

    elif "win" in platform:
        project_dir = re.sub(r'\\server', '', sys.path[0])

    if project_dir not in sys.path:
        sys.path.append(project_dir)
    
    from server.routes import *
    from server.admin_routes import *
    from server import PORT

    with app.app_context():
        db.create_all()
    # debug = False is key for PyTDC to work
    socketio.run(app, debug=False, port=PORT, allow_unsafe_werkzeug=True)