from typing import Any

import numpy as np
import torch
from config.embeddings import BasicEmbeddingConfig, BasicEmbeddingParams
from rdkit import Chem
from src.embeddings.base_embedding import BaseEmbedding
from src.embeddings.const import EDGE_MAPPING
from src.embeddings.embedding_registry import register_embedding
from src.utils.molecules import LeadCompound
from torch_geometric.data import Data


@register_embedding(BasicEmbeddingConfig().name)
class BasicEmbedding(BaseEmbedding):
    """
    Class for basic embedding of lead compounds.

    Args:
        params (BasicEmbeddingParams): Parameters for the basic embedding.
    """

    def __init__(self, params: BasicEmbeddingParams):
        super().__init__(params)
        self.params: BasicEmbeddingParams

    @staticmethod
    def one_of_k_encoding_unk(x: Any, allowable_set: set[Any]):
        """
        Encode a value using one-of-k encoding with an unknown value.

        Args:
            x: The value to encode.
            allowable_set (set): The set of allowable values.

        Returns:
            list: The encoded value.

        """
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    @staticmethod
    def one_of_k_encoding(x: Any, allowable_set: set[Any]):
        """
        Encode a value using one-of-k encoding.

        Args:
            x: The value to encode.
            allowable_set (set): The set of allowable values.

        Returns:
            list: The encoded value.

        Raises:
            ValueError: If the input value is not in the allowable set.

        """
        if x not in allowable_set:
            raise ValueError(
                "input {0} not in allowable set{1}:".format(x, allowable_set)
            )
        return list(map(lambda s: x == s, allowable_set))

    def from_lead_compounds(self, lead_compounds: LeadCompound):
        """
        Generate the embedding for a lead compound.

        Args:
            lead_compounds (LeadCompound): The lead compound.

        Returns:
            Data: The embedding data.

        """
        smiles = lead_compounds.smiles
        mol = Chem.MolFromSmiles(smiles)
        y = lead_compounds.activity

        if y is None:
            y = -1.0  # set any value so that new compounds will be parsed

        if mol is None:
            mol = Chem.MolFromSmiles("")
        if self.params.with_hydrogen:
            mol = Chem.AddHs(mol)
        if self.params.kekulize:
            Chem.Kekulize(mol)

        X = []
        for atom in mol.GetAtoms():
            attributes = []
            attributes += self.one_of_k_encoding_unk(
                atom.GetSymbol(), ["C", "O", "N", "Cl", "F", "S", ""]
            )
            attributes += self.one_of_k_encoding_unk(
                atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
            )
            attributes += self.one_of_k_encoding(
                len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5]
            )
            attributes += self.one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
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
            e.append(EDGE_MAPPING["bond_type"].index(str(bond.GetBondType())))
            e.append(EDGE_MAPPING["stereo"].index(str(bond.GetStereo())))
            e.append(EDGE_MAPPING["is_conjugated"].index(bond.GetIsConjugated()))

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
