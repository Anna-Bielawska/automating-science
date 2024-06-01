"""Standalone server to keep track of and serve the leaderboard"""
from datetime import datetime

from server.app import db


class User(db.Model):
    id = db.Column(db.String(80), primary_key=True)
    oracle_calls = db.Column(db.PickleType, nullable=False)  # Dictionary with {oracle_name: count}
    # compounds = db.Column(db.PickleType, nullable=False) # Dictionary with {target: list of values}
    compound_scores = db.Column(db.PickleType, nullable=False)  # Dictionary with {target: list of values}
    compound_sas_scores = db.Column(db.PickleType, nullable=False)  # Dictionary with {target: list of values}

    # Back reference from Token to User
    tokens = db.relationship('Token', backref='user', lazy=True)

class Result(db.Model):
    user_id = db.Column(db.String(80), primary_key=True)
    metrics = db.Column(db.PickleType, nullable=False)


class Token(db.Model):
    token = db.Column(db.String(128), unique=True, primary_key=True, nullable=False)
    creation_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Foreign Key to link Token to User
    user_id = db.Column(db.String(80), db.ForeignKey('user.id'), nullable=False)

    @classmethod
    def check_valid_token(cls, token_str):
        return cls.query.filter_by(token=token_str).first() is not None
