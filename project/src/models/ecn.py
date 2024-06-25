import torch
from config.models import EdgeConditionedNetworkConfig, EdgeConditionedNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, NNConv, global_max_pool

# code based on: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_nn_conv.py

@register_model(EdgeConditionedNetworkConfig().name)
class ECN(torch.nn.Module):
    def __init__(self, params: EdgeConditionedNetworkParams):
        """
        Edge Conditioned Network model.
        
        Args:
            params (EdgeConditionedNetworkParams): Model parameters.
        """
        super().__init__()
        self.params = params
        self.global_pooling = get_global_pooling(params.global_pooling)
        self.concat_global_pooling = get_global_pooling(params.concat_global_pooling)
        assert len(params.dimensions) == len(params.dropout_rates), \
        f"Different number of hidden layers ({len(params.dimensions)}) and dropout rates ({len(params.dropout_rates)}) was specified."

        # 3 stands for the input edge features dimensionality
        self.convs = torch.nn.ModuleList()
        for i in range(len(params.dimensions)):
            in_dim = params.in_channels if i == 0 else params.dimensions[i - 1]
            nn = MLP([3, 25, in_dim * params.dimensions[i]], norm="batch_norm", dropout=params.dropout_rates[i])
            self.convs.append(
                NNConv(in_dim, params.dimensions[i], nn, aggr='mean')
            )

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
        # Prepare the data
        x, edge_index, edge_attr = (
            data.x,
            data.edge_index,
            data.edge_attr,
        )
        # Apply convolutions
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