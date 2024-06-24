import torch
from config.models import GraphAttentionNetworkConfig, GraphAttentionNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv

# Code based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py

@register_model(GraphAttentionNetworkConfig().name)
class GAT(torch.nn.Module):
    """
    Graph Attention Network model.
    
    Args:
        params (GraphAttentionNetworkParams): Model parameters.
    """
    def __init__(self, params: GraphAttentionNetworkParams):
        super().__init__()
        self.params = params
        self.conv1 = GATConv(
            params.in_channels, params.hidden_channels, params.heads, dropout=0.6
        )
        self.conv2 = GATConv(
            params.hidden_channels * params.heads,
            params.out_channels,
            heads=1,
            concat=False,
            dropout=0.6,
        )
        self.global_pooling = get_global_pooling(params.global_pooling)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data (Batch): The input data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        x, edge_index, edge_attr = (
            x,
            edge_index,
            edge_attr,
        )
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.global_pooling(x, data.batch)
        return x
