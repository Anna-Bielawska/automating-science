"""Spaces of compounds used as baseline in the project."""

from abc import abstractmethod, ABCMeta
from typing import List, Optional
import numpy as np
from tdc.generation import MolGen
from src.utils import RdkitCanonicalSmiles
from rdkit import Chem


class CompoundSpace(metaclass=ABCMeta):
    """Base class for spaces of copounds.

    Notes
    -----
    * Enables sampling narrowed down to a subset. This is key feature for some use cases where
    stratified sampling of the space is needed.
    """

    @abstractmethod
    def try_sample(
            self, **kwargs
    ) -> Optional[List[RdkitCanonicalSmiles]]:
        """
        Attempts to sample a compound from the space, narrowing down to a subset.

        :param reaction_name: If provided, will sample compounds synthesizable using this reaction
        :return None, or list of PotentialSpaceM1Compound

        """
        raise NotImplementedError()

    @abstractmethod
    def estimate_size(self) -> float:
        """Returns the estimate of size of the Space."""
        raise NotImplementedError()


class SmallZINC(CompoundSpace):
    """
    Space that samples compounds from ZINC using a small 250k sample from PyTDC (whole ZINC is >230m).
    """

    def __init__(self, seed: int = 777):
        data = MolGen(name="ZINC")
        self.smiles = data.get_data()["smiles"]
        self.rng = np.random.RandomState(seed)

    def estimate_size(self) -> float:
        return len(self.smiles)

    def try_sample(self, n_molecules=1) -> Optional[List[RdkitCanonicalSmiles]]:
        return self.rng.choice(self.smiles, n_molecules).tolist()

    def sample(self, n_molecules=1) -> Optional[List[RdkitCanonicalSmiles]]:
        return list(map(Chem.MolFromSmiles, self.rng.choice(self.smiles, n_molecules)))
