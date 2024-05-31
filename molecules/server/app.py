"""Standalone server to keep track of and serve the leaderboard"""
import os
from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy

# CONFIGURATION
TOP_N = 10
# normally start with 1 job and let gunicorn handle connections
N_JOBS = int(os.environ.get("N_JOBS", "1"))
SAS_THRESHOLD = 4.0
MASTER_KEY = os.environ.get("MASTER_KEY", "YourSuperSecretMasterKey")
# Define predefined call limits for each oracle_name. Adjust as needed.
# 14.10.2023 choice for workshop
call_limits = { # default is +inf
    "JNK3": 5000,
    # "GSK3β": 5000,
    "DRD2": 1000
}
WORKSHOP_ORACLES = ['DRD2', 'GSK3β', 'JNK3']

# App
app = Flask(__name__)
path = os.path.abspath(os.getcwd())
db_path = os.path.join(path, "results.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
# set unlimited overflow size, otherwise fails
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': N_JOBS,  # too large pool leads to server errors
    'max_overflow': -1  # set to -1 to indicate unlimited overflow
}
db = SQLAlchemy(app)
socketio = SocketIO(app)

def sum_filter(value):
    return sum(value.values())

app.jinja_env.filters['sum'] = sum_filter