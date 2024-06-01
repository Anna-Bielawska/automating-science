"""
Routes of the application:

* /leaderboard: shows up the leaderboard
* /all_results: gets results of all users (same as displayed in the leaderboard)
* /score_compounds_and_update_leaderboard: scores provided compounds and updates the leaderboard
"""
import hashlib
import traceback
from typing import List

import numpy as np
import rdkit
from flask import request, render_template, jsonify
from more_itertools import zip_equal
from rdkit import Chem
from rich.console import Console
from sqlalchemy.orm.attributes import flag_modified

# from port import PORT
from server import PORT
from src.sas_score import compute_ertl_score
from server.app import TOP_N, db, app, call_limits, SAS_THRESHOLD, WORKSHOP_ORACLES, N_JOBS
from server.models import Result, User, Token
from src.eval import virtual_screen_TDC

console = Console()


def _compute_md5(data):
    m = hashlib.md5()
    m.update(str(data).encode('utf-8'))
    return m.hexdigest()

def _validate_smiles(candidates: List[str]):
    """Helper function to check if the SMILES are valid"""
    for s in candidates:
        if not isinstance(s, str):
            raise ValueError("SMILES must be a string.")
        if len(s) == 0:
            raise ValueError("SMILES cannot be empty.")

        try:
            mol = rdkit.Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("Invalid SMILES")
        except Exception as e:
            console.print_exception(show_locals=True)
            raise ValueError(f"Failed to parse SMILES using rdkit: {e}")

def _evaluate_synthesizability(candidates: List[str]) -> List[float]:
    _validate_smiles([c for c in candidates])
    return [compute_ertl_score(c) for c in candidates]

def _get_results_sorted() -> List[Result]:
    results = Result.query.all()
    return list(sorted(results, key=lambda x: -sum(x.metrics.values())))
@app.route('/all_results', methods=['POST', 'GET'])
def all_results():
    sorted_results = _get_results_sorted()
    return jsonify([{"metrics": x.metrics, "user": x.user_id} for x in sorted_results]), 200

@app.route('/leaderboard')
def index():
    sorted_results = _get_results_sorted()
    return render_template('index.html', results=sorted_results, port=PORT)

@app.route("/all_scores", methods=['POST', 'GET'])
def all_scores():
    token = request.json.get('token')

    # Check if the token is valid
    if not Token.check_valid_token(token):
        return jsonify({"error": "Invalid token"}), 403

    token_obj = Token.query.get(token)

    if token_obj is None:
        return jsonify({"error": "Invalid token"}), 403

    user = User.query.get(token_obj.user_id)
    if not user:
        user = User(id=token_obj.user_id, oracle_calls={}, compound_scores={}, compound_sas_scores={})
        db.session.add(user)

    if user.compound_scores is None:
        user.compound_scores = {}
        user.compound_sas_scores = {}
    db.session.commit()

    return jsonify({"compound_scores": user.compound_scores, "compound_sas_scores": user.compound_sas_scores}), 200


@app.route("/score_compounds_and_update_leaderboard", methods=['POST'])
def score_compounds_and_update_leaderboard():
    """
    Scores compounds and updates leaderboard with the running top N score

    Compounds that are too hard to synthesize have returned score -1
    """
    try:
        token = request.json.get('token')
        oracle_name = request.json.get('oracle_name', "_DRD2")
        oracle_name = oracle_name.replace("_server", "")

        if oracle_name not in WORKSHOP_ORACLES:
            return jsonify({"error": f"Expected oracle in {WORKSHOP_ORACLES}"}), 403

        # Check if the token is valid
        if not Token.check_valid_token(token):
            return jsonify({"error": "Invalid token"}), 403

        token_obj = Token.query.get(token)

        if token_obj is None:
            return jsonify({"error": "Invalid token"}), 403

        user = User.query.get(token_obj.user_id)
        if not user:
            user = User(id=token_obj.user_id, oracle_calls={}, compound_scores={}, compound_sas_scores={})
            db.session.add(user)

        oracle_calls = user.oracle_calls

        if oracle_name not in user.oracle_calls:
            oracle_calls[oracle_name] = 0



        n_remaining_calls = call_limits.get(oracle_name, float('inf')) - user.oracle_calls[oracle_name]

        if n_remaining_calls <= 0:
            return jsonify({"error": f"Call limit reached for oracle: {oracle_name}"}), 403

        compounds = request.json.get('compounds')


        if compounds is None:
            return jsonify({"error": "Missing 'compounds' field in the request."}), 500
        compounds = compounds.split(",")
        if len(compounds) > 5000:
            return jsonify({"error": f"Max compounds that can be scored by one call is 5000"}), 403

        if len(compounds) > n_remaining_calls:
            compounds = np.random.RandomState(777).choice(compounds, n_remaining_calls)

        # update the limit
        oracle_calls[oracle_name] += len(compounds)
        user.oracle_calls = oracle_calls # to force sqlalchemy to update the dict
        flag_modified(user, 'oracle_calls')
        db.session.commit()

        user = User.query.get(token_obj.user_id)

        for compound in compounds:
            try:
                mol = rdkit.Chem.MolFromSmiles(compound)

                if mol is None:
                    return jsonify(
                        {"error": f"Failed to parse SMILES {compound} (rdkit.Chem.MolFromSmiles(smi) returns None)."}), 500
            except:
                # Get the traceback details and return it along with the error message
                tb = traceback.format_exc()
                return jsonify({"error": f"Failed to parse SMILES {compound} (rdkit.Chem.MolFromSmiles(smi) throws an error).",
                                "traceback": tb}), 500
        # HACK: replaces "_server" which is special sequence to differntiate DRD2 from DRD2_server
        sas_scores = _evaluate_synthesizability(compounds)
        vs_scores = virtual_screen_TDC(compounds, oracle_name, n_jobs=N_JOBS)
        scores = [vs_score if sas_score <= SAS_THRESHOLD else -1 for vs_score, sas_score in zip_equal(vs_scores, sas_scores)]

        if user.compound_scores is None:
            user.compound_scores = {}
            # user.compounds = {}
            user.compound_sas_scores = {}

        current_compound_score_dict = user.compound_scores
        # current_compounds_dict = user.compounds
        current_compound_sas_score_dict = user.compound_sas_scores

        if oracle_name not in user.compound_scores:
            current_compound_score_dict[oracle_name] = []
            # current_compounds_dict[oracle_name] = []
            current_compound_sas_score_dict[oracle_name] = []

        # current_compounds_dict[oracle_name] += compounds
        current_compound_score_dict[oracle_name] += scores
        current_compound_sas_score_dict[oracle_name] += sas_scores

        # this forces alchemy to commit the change
        user.compound_scores = current_compound_score_dict
        user.compound_sas_scores = current_compound_sas_score_dict
        flag_modified(user, 'compound_scores')
        flag_modified(user, 'compound_sas_scores')

        db.session.commit()

        metrics = {}
        for k in [TOP_N]:
            for oracle_name in WORKSHOP_ORACLES:
                if oracle_name in user.compound_scores:
                    top_ids = np.argsort(user.compound_scores[oracle_name])[-k:]
                    metrics[f"{oracle_name}_top_{k}"] = np.mean([user.compound_scores[oracle_name][i] for i in top_ids])
                else:
                    metrics[f"{oracle_name}_top_{k}"] = 0.0

        result = Result.query.get(user.id)

        if not result:
            result = Result(user_id=user.id, metrics=metrics)
            db.session.add(result)
        else:
            result.metrics = {**result.metrics, **metrics} # this forces alchemy to commit the change
        flag_modified(result, 'metrics') # this forces alchemy to commit the change

        db.session.commit()
        return jsonify({"metrics": metrics, "compound_scores": scores, "compound_sas_scores": sas_scores}), 200

    except Exception as e:
        # Get the traceback details and return it along with the error message
        tb = traceback.format_exc()
        console.log(tb)
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/results-checksum", methods=['GET'])
def results_checksum():
    results = Result.query.all()
    return jsonify({"checksum": _compute_md5(results)})
