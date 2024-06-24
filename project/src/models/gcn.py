import torch
from config.models import GraphNeuralNetworkConfig, GraphNeuralNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
    ):
        """
        Initializes a Graph Convolutional Network (GCN) layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_rate (float): Dropout rate to be applied.
            use_batch_norm (bool): Whether to use batch normalization.

        """
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = (
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
            edge_index (torch.Tensor): Graph edge indices.
            edge_weight (torch.Tensor): Edge weights.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


@register_model(GraphNeuralNetworkConfig().name)
class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network model.

    Args:
        params (GraphNeuralNetworkParams): Model parameters.
    """

    def __init__(
        self,
        params: GraphNeuralNetworkParams,
    ):
        self.params = params
        assert (
            len(params.dimensions) == len(params.dropout_rates)
        ), f"Specified {len(params.dimensions)} convolutional layers, but {len(params.dropout_rates)} dropout rates "

        super().__init__()
        self.global_pooling = get_global_pooling(params.global_pooling)
        self.concat_global_pooling = get_global_pooling(params.concat_global_pooling)

        self.layers = nn.ModuleList()

        for i in range(len(params.dimensions)):
            in_dim = params.input_dim if i == 0 else params.dimensions[i - 1]
            self.layers.append(
                GCNLayer(in_dim, params.dimensions[i], params.dropout_rates[i])
            )

        self.out_layer = (
            torch.nn.Linear(2 * params.dimensions[-1], params.output_dim)
            if self.concat_global_pooling is not None
            else torch.nn.Linear(params.dimensions[-1], params.output_dim)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Args:
            data (Batch): The input data.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Prepare the data
        x, edge_index, edge_weight = (
            data.x,
            data.edge_index,
            data.edge_weight,
        )

        # Apply convolutional layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)

        # Apply global pooling
        _x = x
        x = self.global_pooling(x, data.batch)

        # Concatenate global pooling
        if self.concat_global_pooling:
            _x = self.concat_global_pooling(_x, data.batch)
            x = torch.cat([x, _x], dim=-1)

        # Apply final linear layere
        x = self.out_layer(x)

        return x
