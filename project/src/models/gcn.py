import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from src.models.model_registry import register_model
from config.models import GraphNeuralNetworkConfig


@register_model(GraphNeuralNetworkConfig().name)
class GraphNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        dimensions: list[int],
        dropout_rates: list[float],
        input_dim: int = 100,
        output_dim: int = 1,
        concat_max_pooling: bool = False,
    ):
        assert (
            len(dimensions) == len(dropout_rates)
        ), f"Specified {len(dimensions)} convolutional layers, but {len(dropout_rates)} dropout rates "

        super().__init__()

        layers_num = len(dimensions)
        self.concat_max_pooling = concat_max_pooling

        self.conv_layers = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=dimensions[0])]
            + [
                GCNConv(in_channels=dimensions[i], out_channels=dimensions[i + 1])
                for i in range(0, layers_num - 1)
            ]
        )

        self.drop_layers = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout_rates[j]) for j in range(len(self.conv_layers))]
        )

        self.batch_norms = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=dimensions[j])
                for j in range(len(self.conv_layers))
            ]
        )

        self.out_layer = (
            torch.nn.Linear(2 * dimensions[-1], output_dim)
            if self.concat_max_pooling
            else torch.nn.Linear(dimensions[-1], output_dim)
        )

    def forward(self, data: Batch) -> torch.Tensor:
        # NOTE: GCNs do not use edge_attributes, which may be used in some other GNN layers

        # Prepare the data
        x, edge_index, edge_weight = (
            data.x,
            data.edge_index,
            data.edge_weight,
        )

        # Apply convolutional layers
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = self.drop_layers[i](x)
            x = torch.functional.F.relu(x, inplace=False)

        # Apply mean pooling
        _x = x
        x = global_mean_pool(x, data.batch)

        # Concatenate max pooling
        if self.concat_max_pooling:
            _x = global_max_pool(_x, data.batch)
            x = torch.cat([x, _x], dim=-1)

        # Apply final linear layere
        x = self.out_layer(x)

        return x
