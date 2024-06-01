"""Simple wrapper for app"""
import json
from rich.console import Console
from typing import List
import requests

from src.utils import Oracles

class FlaskAppClient:
    ERROR_KEY = "error"
    TRACEBACK_KEY = "traceback"

    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.console = Console()

    def _handle_response(self, response):
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            self.console.print("[red]Failed to parse server response as JSON[/red]")
            self.console.print("Response from server: " + str(response))
            response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code.

        if response.status_code == 200:
            return response_data
        else:
            error = response_data.get(self.ERROR_KEY, 'Unknown error')
            tb = response_data.get(self.TRACEBACK_KEY, None)
            self.console.print(f"[red]Server error: {error}[/red]")
            if tb:
                self.console.print(f"[yellow]{tb}[/yellow]")
            raise RuntimeError(f"Server error: {error}")

    def all_results(self):
        response = requests.post(f"{self.base_url}/all_results", json={})
        return self._handle_response(response)

    def all_scores(self, user_token):
        payload = {
            "token": user_token
        }
        response = requests.post(f"{self.base_url}/all_scores", json=payload)
        return self._handle_response(response)

    def score_compounds_and_update_leaderboard(self, compounds, oracle_name, user_token):
        payload = {
            "compounds": ",".join(compounds),
            "oracle_name": oracle_name,
            "token": user_token
        }
        response = requests.post(f"{self.base_url}/score_compounds_and_update_leaderboard", json=payload)
        return self._handle_response(response)

# Usage Example:
if __name__ == "__main__":
    client = FlaskAppClient()
    token = "test-0"

    # Example for scoring compounds
    compounds = ["CC", "CCC"]
    oracle_name = "DRD2"
    response = client.score_compounds_and_update_leaderboard(compounds, oracle_name, token)
    print(response)

    # Example of error handling
    compounds = ["Cxxxxx"]
    oracle_name = "DRD2"
    response = client.score_compounds_and_update_leaderboard(compounds, oracle_name, token)
    print(response)