from config.embeddings.basic_embedding import BaseEmbeddingConfig
from utils.molecules import LeadCompound


class BaseEmbedding:
    def __init__(self, config: BaseEmbeddingConfig):
        self.config = config

    def from_lead_compounds(self, lead_compounds: LeadCompound):
        raise NotImplementedError