import random
from src.utils.mutate import mutate_selfie
from src.utils.molecules import LeadCompound, compute_ertl_score
from src.loops.base_loop import Loop
from pathlib import Path
import selfies


class MutateLoop(Loop):
    """Implementation of AL algorithm that mutates top compounds from the previous iterations.

    Mutate loop should first search random and then mutate top compounds
    """

    def __init__(
        self,
        initial_dataset: list[LeadCompound],
        base_dir: Path,
        n_warmup_iterations: int = 1,
        mutate_top_k: int = 10,
        user_token=None,
        target="DRD2",
    ):
        self.initial_dataset = initial_dataset
        self.n_warmup_iterations = n_warmup_iterations
        self.mutate_top_k = mutate_top_k
        super().__init__(base_dir, user_token, target)

    def _propose_random(self, n_candidates: int) -> list[LeadCompound]:
        return random.sample(self.initial_dataset, k=n_candidates)

    def propose_candidates(self, n_candidates: int) -> list[LeadCompound]:
        if n_candidates < self.mutate_top_k:
            raise ValueError(
                f"n_candidates must be at least mutate_top_k ({self.mutate_top_k})."
            )

        if n_candidates == 0:
            return []

        if self.n_iterations < self.n_warmup_iterations:
            return self._propose_random(n_candidates)

        previous_results: list[LeadCompound] = self.load()
        candidates = list(
            sorted(previous_results, key=lambda m: (-m.activity, -m.synth_score))
        )
        selfie_candidates = [
            selfies.encoder(m.smiles) for m in candidates[: self.mutate_top_k]
        ]

        new_compounds = []
        while len(set(new_compounds)) < n_candidates:
            for selfie in selfie_candidates:
                new_selfie = selfies.decoder(
                    mutate_selfie(selfie, max_molecules_len=100)[0]
                )
                if (
                    compute_ertl_score(new_selfie) > 1.0
                    and compute_ertl_score(new_selfie) < 4.0
                ):
                    new_compounds.append(new_selfie)
                if len(set(new_compounds)) == n_candidates:
                    break

        new_compounds = set(new_compounds)

        assert len(new_compounds) == n_candidates
        return [LeadCompound(smiles=c) for c in new_compounds]
