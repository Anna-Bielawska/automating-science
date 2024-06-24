import torch
from config.models import GraphIsomorphismNetworkConfig, GraphIsomorphismNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, GINConv, global_max_pool

# code based on: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mutag_gin.py

@register_model(GraphIsomorphismNetworkConfig().name)
class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network model.
    
    Args:
        params (GraphIsomorphismNetworkParams): Model parameters.
    """
    def __init__(self, params: GraphIsomorphismNetworkParams):
        super().__init__()
        self.params = params
        self.in_channels = params.in_channels  # initialize the input channels dim, it will be changed
        self.global_pooling = get_global_pooling(params.global_pooling)
        self.concat_global_pooling = get_global_pooling(params.concat_global_pooling)

        self.convs = torch.nn.ModuleList()
        for _ in range(params.num_layers):
            mlp = MLP([self.in_channels, params.hidden_channels],
                      norm="batch_norm", dropout=0.3)
            self.convs.append(GINConv(nn=mlp, train_eps=params.trainable_eps))
            self.in_channels = params.hidden_channels  # update the input channels dim

        # simple FC on top for predictions
        if self.concat_global_pooling:
            self.mlp = MLP([2*params.hidden_channels, params.out_channels],
                           norm="batch_norm", dropout=0.3)
        else:
            self.mlp = MLP([params.hidden_channels, params.out_channels],
                           norm="batch_norm", dropout=0.3)

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data (Batch): The input data.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        # Prepare the data
        x, edge_index = (
            data.x,
            data.edge_index
        )
        # Apply convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x, inplace=False)
        # Apply mean pooling
        _x = x
        x = self.global_pooling(x, data.batch)
        # Concatenate max pooling
        if self.concat_global_pooling:
            _x = self.concat_global_pooling(_x, data.batch)
            x = torch.cat([x, _x], dim=-1)
        return self.mlp(x)
