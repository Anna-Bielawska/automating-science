import torch
from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential


# from https://github.com/pyg-team/pytorch_geometric/blob/dafbd3013b19737ac1511d16d22d6529786a63c4/examples/pna.py#L8
class PNA(torch.nn.Module):
    def __init__(self, indegree_histogram: torch.Tensor):
        super().__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(
                in_channels=75,
                out_channels=75,
                aggregators=aggregators,
                scalers=scalers,
                deg=indegree_histogram,
                edge_dim=50,
                towers=5,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(
            Linear(75, 50), ReLU(), Linear(50, 25), ReLU(), Linear(25, 1)
        )
        self.relu = ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): Edge index tensor.
            edge_attr (torch.Tensor): Edge attribute tensor.
            batch (torch.Tensor): Batch tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """

        x = self.node_emb(x.squeeze())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)
