import torch
from config.models import GraphAttentionNetworkConfig, GraphAttentionNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, GATConv

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
        self.global_pooling = get_global_pooling(params.global_pooling)
        self.concat_global_pooling = get_global_pooling(params.concat_global_pooling)

        assert params.heads[-1] == 1, f"Last GAT layer must have 1 attention head, but {params.heads[-1]} were specified"
        assert len(params.dimensions) == len(params.dropout_rates), \
        f"Different number of hidden layers ({len(params.dimensions)}) and dropout rates ({len(params.dropout_rates)}) was specified."
        print(params.dimensions, params.heads)
        assert len(params.heads) == len(params.dimensions), "Mismatch in the number of attention layers and corresponding heads"

        self.convs = torch.nn.ModuleList()
        for i in range(len(params.dimensions)):
            in_dim = params.in_channels if i == 0 else params.dimensions[i - 1]
            concat = False if i == len(params.dimensions)-1 else True
            prev_heads = 1 if i == 0 else params.heads[i-1]

            self.convs.append(GATConv(
                in_dim * prev_heads,
                params.dimensions[i],
                heads=params.heads[i],
                concat=concat,
                dropout=params.dropout_rates[i]
            ))
        
        if self.concat_global_pooling:
            self.fc = MLP([2*params.dimensions[-1], params.out_channels],
                           norm="batch_norm", dropout=0.3)
        else:
            self.fc = MLP([params.dimensions[-1], params.out_channels],
                           norm="batch_norm", dropout=0.3)

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
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
        # Apply global average pooling
        _x = x
        x = self.global_pooling(x, data.batch)
        # Concatenate global max pooling
        if self.concat_global_pooling:
            _x = self.concat_global_pooling(_x, data.batch)
            x = torch.cat([x, _x], dim=-1)
        return self.fc(x)
