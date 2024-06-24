import copy
import glob
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

import rdkit
from config.loops import BaseLoopParams
from config.main_config import TrainConfig
from more_itertools import zip_equal
from src.utils.molecules import LeadCompound, compute_ertl_score
from src.utils.screening import run_virtual_screening

logger = logging.getLogger(__name__)

SAS_THRESHOLD = 4.0


class BaseLoop:
    """Base class for AL loop"""

    def __init__(
        self,
        loop_params: BaseLoopParams,
        base_dir: Union[str, Path],
        target="GSK3β",
        training_cfg: TrainConfig = None,
    ):
        """
        Initialize the BaseLoop object.

        Args:
            loop_params (BaseLoopParams): Parameters for the active learning loop.
            base_dir (Union[str, Path]): Base directory to store the results.
            target (str, optional): Target for the active learning loop. Defaults to "GSK3β".
            training_cfg (TrainConfig, optional): Training configuration. Defaults to None.
        """
        self.loop_params = loop_params
        self.base_dir = base_dir if isinstance(base_dir, Path) else Path(base_dir)
        self.training_cfg = training_cfg
        self.target = target

        logger.debug(f"The results will be stored in {self.base_dir}")
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True)

    @abstractmethod
    def propose_candidates(self, n_candidates: int) -> list[LeadCompound]:
        """
        A stateful function that proposes candidates based on prior experience.

        Args:
            n_candidates (int): Number of candidates to propose.

        Returns:
            list[LeadCompound]: List of proposed lead compounds.
        """
        pass

    @classmethod
    def evaluate_synthesizability(cls, candidates: list[LeadCompound]) -> list[float]:
        """
        Evaluate the synthesizability of the given candidates.

        Args:
            candidates (list[LeadCompound]): List of lead compounds to evaluate.

        Returns:
            list[float]: List of synthesizability scores for the candidates.
        """
        cls._validate_smiles([c.smiles for c in candidates])
        return [compute_ertl_score(c.smiles) for c in candidates]

    @property
    def n_iterations(self) -> int:
        """
        Get the number of iterations.

        Returns:
            int: Number of iterations.
        """
        return len(list(glob.glob(str(self.base_dir / "*.json"))))

    def load(self, iteration_id: Optional[int] = None) -> list[LeadCompound]:
        """
        Load the results of previous iterations from the base_dir.
        If iteration_id is None, then load all results.

        Args:
            iteration_id (Optional[int], optional): Iteration ID to load results from. Defaults to None.

        Returns:
            list[LeadCompound]: List of lead compounds loaded from the results.
        """
        all_res = list(glob.glob(str(self.base_dir / "*.json")))
        # sort by index (filenames are like 0.json, 1.json, 2.json, ...)
        all_res.sort(key=lambda x: int(Path(x).stem))
        if iteration_id is not None:
            c = json.load(open(all_res[iteration_id], "r"))
        else:
            c = sum([json.load(open(f, "r")) for f in all_res], [])
        return list(map(LeadCompound.from_dict, c))

    def test_in_lab_and_save(
        self, candidates: list[LeadCompound]
    ) -> list[LeadCompound]:
        """
        Test candidates in the lab and save the outcome locally.

        The compounds are first checked for synthesizability using synthesize() function.
        The results are saved in base_dir/[date]_lab_results.json.

        Args:
            candidates (list[LeadCompound]): List of lead compounds to test.

        Returns:
            list[LeadCompound]: List of lead compounds with updated activity and synthesis scores.
        """
        candidates = copy.deepcopy(candidates)

        smi = [c.smiles for c in candidates]
        if len(set(smi)) != len(smi):
            raise ValueError("Duplicate SMILES detected.")

        self._validate_smiles([c.smiles for c in candidates])
        # try to synthesize
        synthesizability_scores = self.evaluate_synthesizability(candidates)
        # compute scores (NOTE: implemented this way to be seamless for the user)
        if self.target == "GSK3β":
            # this target is evaluated locally as it has unlimited # of calls
            metrics, activity_scores = run_virtual_screening(
                [c.smiles for c in candidates], self.target
            )
            for c, a_score, s_score in zip_equal(
                candidates, activity_scores, synthesizability_scores
            ):
                if s_score <= SAS_THRESHOLD:
                    c.activity = a_score
                else:
                    c.activity = -1
                c.synth_score = s_score
        else:
            raise NotImplementedError(f"Target {self.target} is not implemented.")

        # save results
        save_filename = "{}.json".format(self.n_iterations)
        logger.info(f"Saving results to {self.base_dir / save_filename}.")
        json.dump(
            [c.to_dict() for c in sorted(candidates, key=lambda x: x.smiles)],
            open(self.base_dir / save_filename, "w"),
            indent=2,
        )

        return candidates

    @classmethod
    def _validate_smiles(cls, candidates: list[str]) -> None:
        """
        Helper function to check if the SMILES are valid.

        Args:
            candidates (list[str]): List of SMILES strings.

        Raises:
            ValueError: If the SMILES are invalid.
        """
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
                logger.error(f"Failed to parse SMILES using rdkit: {e}")
                raise ValueError(f"Failed to parse SMILES using rdkit: {e}")
