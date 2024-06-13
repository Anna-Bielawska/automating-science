from dataclasses import dataclass
from dataclasses_json import dataclass_json
from rdkit import Chem
from src.utils.sas_score import calculateScore
import numpy as np
from torch_geometric.data import Data
import torch
from typing import Any, Optional


@dataclass_json
@dataclass
class LeadCompound:
    """Dataclass for a lead compound, which can have outcomes of synthesizability and activity."""

    smiles: str
    synth_score: Optional[float] = None
    activity: Optional[float] = None


def compute_ertl_score(compound: str) -> float:
    """Compute Ertl (Synthetic Accessibility) score for a compound.

    Args:
        compound (str): SMILES representation of the compound for which to compute the Ertl score.

    Returns:
        float: The Ertl score of the compound, indicating its synthetic accessibility. Lower scores correspond to compounds that are easier to synthesize.
    """

    mol = Chem.MolFromSmiles(compound)
    return calculateScore(mol)


# Based on https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/utils/smiles.html


def one_of_k_encoding_unk(x, allowable_set: set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set: set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


E_MAP: dict[str, list[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


def from_lead_compound(
    compound: LeadCompound, with_hydrogen: bool = False, kekulize: bool = False
) -> Data:
    """Converts a LeadCompound to a :class:`torch_geometric.data.Data` instance.

    Args:
        compound (LeadCompound): LeadCompound object.
        with_hydrogen (bool, optional): Should store hydrogens in the graph. Defaults to False.
        kekulize (bool, optional): Should convert aromatic bonds to single/double bonds. Defaults to False.

    Returns:
        torch_geometric.data.Data: Torch Geometric Data object with edge_weight and smiles atrributes.
    """
    r"""Converts a LeadCompound to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        compound (LeadCompound): The LeadCompound object.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """

    smiles = compound.smiles
    mol = Chem.MolFromSmiles(smiles)
    y = compound.activity

    if y is None:
        y = -1.0  # set any value so that new compounds will be parsed

    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    X = []
    for atom in mol.GetAtoms():
        attributes = []
        attributes += one_of_k_encoding_unk(
            atom.GetSymbol(), ["C", "O", "N", "Cl", "F", "S", ""]
        )
        attributes += one_of_k_encoding_unk(
            atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        )
        attributes += one_of_k_encoding(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
        attributes += one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        attributes.append(atom.GetFormalCharge())
        attributes.append(atom.IsInRing())
        attributes.append(atom.GetIsAromatic())
        attributes.append(atom.GetExplicitValence())
        attributes.append(atom.GetNumRadicalElectrons())
        attributes = np.array(attributes).astype(float)
        X.append(attributes)

    X = np.stack(X)
    x = torch.tensor(X, dtype=torch.float32)

    edge_indices, edge_attrs, edge_weights = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        val = mol.GetBondBetweenAtoms(i, j).GetBondTypeAsDouble()

        e = []
        e.append(E_MAP["bond_type"].index(str(bond.GetBondType())))
        e.append(E_MAP["stereo"].index(str(bond.GetStereo())))
        e.append(E_MAP["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]
        edge_weights += [val, val]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, 3)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr, edge_weight = (
            edge_index[:, perm],
            edge_attr[perm],
            edge_weight[perm],
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        y=y,
        edge_weight=edge_weight,
    )
