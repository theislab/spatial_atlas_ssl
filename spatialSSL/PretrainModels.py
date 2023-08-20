from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv


class GraphSequential(nn.Module):
    def __init__(self, *layers):
        super(GraphSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights).relu()
        return x


class GCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hidden_layers=1, num_classes=None):
        """
        :param in_channels: number of input features
        :param hidden_channels: number of hidden features
        :param num_hidden_layers: number of hidden layers
        :param num_classes: number of output classes, if set to None, model can be used to classify nodes
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers

        layers = []
        for i in range(num_hidden_layers + 1):
            if i == 0:
                layers.append(GCNConv(in_channels, hidden_channels))
            else:
                layers.append(GCNConv(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())

        self.layers = GraphSequential(*layers)

        if num_classes is None:
            self.final_layer_pre = GCNConv(hidden_channels, in_channels)
        else:
            self.final_layer_down = GCNConv(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        x = self.layers(x, edge_index, edge_weights)
        x = self.final_layer(x, edge_index, edge_weights)
        return x


"""class GCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layer1 = GCNConv(in_channels, hidden_channels)
        self.layer2 = GCNConv(hidden_channels, hidden_channels)
        self.layer3 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        x = self.layer1(x, edge_index, edge_weights).relu()
        x = self.layer2(x, edge_index, edge_weights).relu()
        x = self.layer3(x, edge_index, edge_weights)
        return x
        
"""
