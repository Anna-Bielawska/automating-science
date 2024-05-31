import rdkit
from rdkit import Chem
from tdc import Oracle
from src.al_loop import LeadCompound


class HighThroughputScreening:
    def __init__(self):
        ...

    def test_molecules(self, molecules: Chem.Mol, target: str = "GSK3β"):
        oracle = Oracle(name=target)
        smileses = [Chem.MolToSmiles(mol) for mol in molecules]
        preds = oracle(smileses)
        return [LeadCompound(smiles=smiles, activity=pred) for smiles, pred in zip(smileses, preds)]