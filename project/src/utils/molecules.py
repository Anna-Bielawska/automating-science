from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json
from rdkit import Chem
from src.utils.sas_score import calculateScore


@dataclass_json
@dataclass
class LeadCompound:
    """Dataclass for a lead compound, which can have outcomes of synthesizability and activity."""

    smiles: str
    synth_score: Optional[float] = None
    activity: Optional[float] = None

    def __hash__(self):
        return hash(self.smiles)


def compute_ertl_score(compound: str) -> float:
    """Compute Ertl (Synthetic Accessibility) score for a compound.

    Args:
        compound (str): SMILES representation of the compound for which to compute the Ertl score.

    Returns:
        float: The Ertl score of the compound, indicating its synthetic accessibility. Lower scores correspond to compounds that are easier to synthesize.
    """

    mol = Chem.MolFromSmiles(compound)
    return calculateScore(mol)

