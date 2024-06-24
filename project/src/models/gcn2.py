import torch
from config.models import GraphResidualNetworkConfig, GraphResidualNetworkParams
from src.models.model_registry import register_model
from src.utils.graph import get_global_pooling
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCN2Conv


class GCN2Layer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0,
        use_batchnorm: bool = False,
        alpha: float = 0.5,
        theta: float = 1.0,
        layer: int = 0
    ):
        """
        Initializes a residual Graph Convolutional Network (GCN) layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout_rate (float): Dropout rate to be applied.
            use_batch_norm (bool): Whether to use batch normalization.
            alpha (float): The strength of the initial residual connection.
            theta (float): The hyperparameter to compute the strength of the identity mapping.
            layer (int): The layer in which this module is executed (needed for the math underneath).

        """
        super(GCN2Layer, self).__init__()
        self.conv = GCN2Conv(
            in_channels,
            out_channels,
            alpha=alpha,
            theta=theta,
            shared_weights=True, # will use the same weight matrices for the smoothed representation and the initial residual
            layer=layer+1

        )
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = (
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, x_0: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
            x_0 (torch.Tensor): Initial features tensor.
            edge_index (torch.Tensor): Graph edge indices.
            edge_weight (torch.Tensor): Edge weights.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = F.dropout(x, self.dropout, training=self.training)
        h = conv(h, x_0, edge_index, edge_weight)
        x = h + x
        x = self.batch_norms(x)
        x = self.relu(x)
        return x


@register_model(GraphResidualNetworkConfig().name)
class GraphResidualNetwork(nn.Module):
    """
    Graph Residual Network model.

    Args:
        params (GraphResidualNetworkParams): Model parameters.
    """

    def __init__(
        self,
        params: GraphResidualNetworkParams,
    ):
        self.params = params
        assert (
            len(params.dimensions) == len(params.dropout_rates)
        ), f"Specified {len(params.dimensions)} convolutional layers, but {len(params.dropout_rates)} dropout rates "

        super().__init__()
        self.global_pooling = get_global_pooling(params.global_pooling)
        self.concat_global_pooling = get_global_pooling(params.concat_global_pooling)

        self.layers = nn.ModuleList()
        self.in_layer = Linear(params.input_dim, params.dimensions[0])

        for i in range(len(params.dimensions)):
            in_dim = params.input_dim if i == 0 else params.dimensions[i - 1]
            self.layers.append(
                GCN2Layer(params.dimensions[i], params.dimensions[i], params.dropout_rates[i], layer=i)
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

        x_0 = x = self.in_layer(x)
        # Apply convolutional layers
        for layer in self.layers:
            x = layer(x, x_0, edge_index, edge_weight)

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
