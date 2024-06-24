from config.embeddings import BaseEmbeddingParams
from src.utils.molecules import LeadCompound


class BaseEmbedding:
    """Base class for embeddings."""
    def __init__(self, params: BaseEmbeddingParams):
        self.params = params

    def from_lead_compounds(self, lead_compounds: LeadCompound):
        """Converts a LeadCompound to a :class:`torch_geometric.data.Data` instance."""
        raise NotImplementedError