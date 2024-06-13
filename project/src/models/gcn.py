import torch
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GraphNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        model_dim: list[int],
        dropout_rate: list[float],
        input_dim: int = 100,
        output_dim: int = 1,
        concat_max_pooling: bool = False,
    ):
        assert (
            len(model_dim) == len(dropout_rate)
        ), f"Specified {len(model_dim)} convolutional layers, but {len(dropout_rate)} dropout rates "

        super().__init__()

        self.model_dim = model_dim
        self.layers_num = len(model_dim)
        self.dropout_rate = dropout_rate
        self.concat_max_pooling = concat_max_pooling

        self.conv_layers = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=self.model_dim[0])]
            + [
                GCNConv(
                    in_channels=self.model_dim[i], out_channels=self.model_dim[i + 1]
                )
                for i in range(0, self.layers_num - 1)
            ]
        )

        self.drop_layers = torch.nn.ModuleList(
            [
                torch.nn.Dropout(p=self.dropout_rate[j])
                for j in range(len(self.conv_layers))
            ]
        )

        self.batch_norms = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(num_features=self.model_dim[j])
                for j in range(len(self.conv_layers))
            ]
        )

        self.out_layer = (
            torch.nn.Linear(2 * model_dim[-1], output_dim)
            if self.concat_max_pooling
            else torch.nn.Linear(model_dim[-1], output_dim)
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
