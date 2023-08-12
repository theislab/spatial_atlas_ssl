from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv


class GCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layer1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        x = self.layer1(x, edge_index, edge_weights).relu()
        x = self.conv2(x, edge_index, edge_weights)
        return x
