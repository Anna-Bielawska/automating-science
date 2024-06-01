"""Base class for AL loop.

It contains most functions to be used by workshop participants.
"""
import copy
import glob
import json
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Union
from dataclasses import dataclass
from dataclasses_json import dataclass_json

import rdkit
from more_itertools import zip_equal
from rich.console import Console

from src.visualize_compounds import generate_html_molgrid
from src.eval import run_virtual_screening
from src.sas_score import compute_ertl_score
from src.server_wrapper import FlaskAppClient

console = Console()

SAS_THRESHOLD = 4.0

@dataclass_json
@dataclass
class LeadCompound():
    """Dataclass for a lead compound, which can have outcomes of synthesizability and activity."""
    smiles: str
    synth_score: Optional[float] = None
    activity: Optional[float] = None

class Loop():
    """Base class for AL loop"""
    def __init__(self, base_dir: Union[str, Path], user_token: Optional[str] = None, target="DRD2"):
        """
        :param base_dir: directory where the results will be stored
        :param user_token: token used for the user (each user has up to some limit of calls for each target)
        :param target: target for the virtual screening (DRD2, DRD2_server, ...)
        """
        self.base_dir = base_dir if isinstance(base_dir, Path) else Path(base_dir)
        console.log(f"Saving results to {self.base_dir}.")
        self.target = target
        self.user_token = user_token
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True)

    @abstractmethod
    def propose_candidates(self, n_candidates: int) -> List[LeadCompound]:
        """A stateful function that proposes candidates based on prior experience"""
        pass

    @classmethod
    def evaluate_synthesizability(cls, candidates: List[LeadCompound]) -> List[float]:
        cls._validate_smiles([c.smiles for c in candidates])
        return [compute_ertl_score(c.smiles) for c in candidates]

    def generate_visualization(self, iteration_id: Optional[int] = None):
        """Visualize the results of previous iterations in a grid plot."""
        compounds = self.load(iteration_id)
        smiles = [c.smiles for c in compounds]
        properties = {k:  [getattr(c, k) for c in compounds] for k in ['synth_score', 'activity']}
        save_path = self.base_dir / f"visualize_{iteration_id}.html"
        console.log(f"Saving visualization to {save_path} for {len(smiles)} compounds from iteration {iteration_id}.")
        generate_html_molgrid(smiles, str(save_path), properties=properties)

    @property
    def n_iterations(self):
        return len(list(glob.glob(str(self.base_dir / "*.json"))))

    def load(self, iteration_id: Optional[int] = None) -> List[LeadCompound]:
        """Load the results of previous iterations from the base_dir.
        If iteration_id is None, then load all results."""
        all_res = list(glob.glob(str(self.base_dir / "*.json")))
        # sort by index (filenames are like 0.json, 1.json, 2.json, ...)
        all_res.sort(key=lambda x: int(Path(x).stem))
        if iteration_id is not None:
            c = json.load(open(all_res[iteration_id], "r"))
        else:
            c = sum([json.load(open(f, "r")) for f in all_res], [])
        return list(map(LeadCompound.from_dict, c))

    def test_in_lab_and_save(self, candidates: List[LeadCompound], client: Optional[FlaskAppClient]=None):
        """Test candidates in the lab and saves the outcome locally.

         The compounds are first checked for synthesizability using synthesize() function.

        The results are saved in base_dir/[date]_lab_results.json."""
        candidates = copy.deepcopy(candidates)

        smi = [c.smiles for c in candidates]
        if len(set(smi)) != len(smi):
            raise ValueError("Duplicate SMILES detected.")

        self._validate_smiles([c.smiles for c in candidates])
        # try to synthesize
        synthesizability_scores = self.evaluate_synthesizability(candidates)
        # compute scores (NOTE: implemented this way to be seamless for the user)
        if self.target == "GSK3β": # DRD2 was previous choice
            # this target is evaluated locally as it has unlimited # of calls
            metrics, activity_scores = run_virtual_screening([c.smiles for c in candidates], self.target)
            for c, a_score, s_score in zip_equal(candidates, activity_scores, synthesizability_scores):
                if s_score <= SAS_THRESHOLD:
                    c.activity = a_score
                else:
                    c.activity = -1
                c.synth_score = s_score
        else:
            if self.user_token is None:
                raise ValueError("Please provide user_token to test in the lab.")
            response = client.score_compounds_and_update_leaderboard([c.smiles for c in candidates], self.target, self.user_token)
            for c, a_score, s_score in zip_equal(candidates, response['compound_scores'], response['compound_sas_scores']):
                c.activity = a_score
                c.synth_score = s_score

        # save results
        save_filename = "{}.json".format(self.n_iterations)
        console.log(f"Saving results to {self.base_dir / save_filename}.")
        json.dump([c.to_dict() for c in candidates],
                  open(self.base_dir / save_filename, "w"), indent=2)

        return candidates
    @classmethod
    def _validate_smiles(cls, candidates: List[str]):
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