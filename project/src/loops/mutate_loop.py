import random
from pathlib import Path

import selfies
from config.loops import MutateLoopParams
from src.loops.base_loop import BaseLoop
from src.utils.molecules import LeadCompound, compute_ertl_score
from src.utils.mutate import SelfieMutator


class MutateLoop(BaseLoop):
    """
    Loop that proposes new candidates by mutating the top k compounds.

    Args:
        loop_params (MutateLoopParams): Loop parameters.
        initial_dataset (list[LeadCompound]): Initial dataset of lead compounds.
        base_dir (Path): Base directory.
        target (str): Target to optimize.
        seed (int): Random seed.
    """

    def __init__(
        self,
        loop_params: MutateLoopParams,
        initial_dataset: list[LeadCompound],
        base_dir: Path,
        target: str = "GSK3β",
        seed: int = 0,
    ):
        super().__init__(loop_params, base_dir, target)
        self.loop_params: MutateLoopParams
        self.initial_dataset = initial_dataset
        self.selfie_mutator = SelfieMutator(max_workers=loop_params.n_workers, seed=seed)

    def _propose_random(self, n_candidates: int) -> list[LeadCompound]:
        """
        Proposes a random selection of lead compounds from the initial dataset.

        Args:
            n_candidates (int): The number of candidates to propose.

        Returns:
            list[LeadCompound]: A list of randomly selected lead compounds.
        """
        return random.sample(self.initial_dataset, k=n_candidates)

    def propose_candidates(self, n_candidates: int) -> list[LeadCompound]:
        """
        Proposes a list of candidate compounds for mutation.

        Args:
            n_candidates (int): The number of candidate compounds to propose.

        Returns:
            list[LeadCompound]: A list of LeadCompound objects representing the proposed candidate compounds.

        Raises:
            ValueError: If n_candidates is less than mutate_top_k.

        """
        if n_candidates < self.loop_params.mutate_top_k:
            raise ValueError(
                f"n_candidates must be at least mutate_top_k ({self.loop_params.mutate_top_k})."
            )

        if n_candidates == 0:
            return []

        if self.n_iterations < self.loop_params.n_warmup_iterations:
            return self._propose_random(n_candidates)

        previous_results: list[LeadCompound] = self.load()
        candidates = list(
            sorted(previous_results, key=lambda m: (-m.activity, -m.synth_score))
        )
        selfie_candidates = [
            selfies.encoder(m.smiles)
            for m in candidates[: self.loop_params.mutate_top_k]
        ]

        new_compounds = []
        while len(set(new_compounds)) < n_candidates:
            new_selfies = self.selfie_mutator.parallel_mutate_selfies(
                selfie_candidates, max_molecules_len=100
            )
            for new_selfie in new_selfies:
                decoded_selfie = selfies.decoder(new_selfie[0])
                if 1.0 < compute_ertl_score(decoded_selfie) < 4.0:
                    if len(set(new_compounds)) < n_candidates:
                        new_compounds.append(decoded_selfie)
                    else:
                        break

        new_compounds = set(new_compounds)

        assert len(new_compounds) == n_candidates
        return sorted(
            [LeadCompound(smiles=c) for c in new_compounds], key=lambda x: x.smiles
        )
