from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GATConv


class GraphSequential(nn.Module):
    def __init__(self, *layers):
        super(GraphSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index, edge_weights=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weights).relu() #
        return x


class GCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hidden_layers=1, type="GCN", num_classes=None):
        """
        :param in_channels: number of input features
        :param hidden_channels: number of hidden features
        :param num_hidden_layers: number of hidden layers
        :param num_classes: number of output classes, if set to None, model can be used to classify nodes
        """
        super().__init__()

        self.num_hidden_layers = num_hidden_layers
        self.num_classes = num_classes
        self.type = type

        if type == "GCN":
            layer = GCNConv
        elif type == "GAT":
            layer = GATConv
        else:
            raise ValueError("type must be either GCN or GAT")

        layers = []
        for i in range(num_hidden_layers + 1):
            if i == 0:
                layers.append(layer(in_channels, hidden_channels))
            else:
                layers.append(layer(hidden_channels, hidden_channels))
            #layers.append(nn.ReLU())

        self.layers = GraphSequential(*layers)

        if num_classes is None:
            self.final_layer_pre = layer(hidden_channels, in_channels)
        else:
            self.final_layer_down = layer(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weights: Tensor = None) -> Tensor:
        x = self.layers(x, edge_index, edge_weights)

        if self.num_classes is None:
            x = self.final_layer_pre(x, edge_index, edge_weights)
        else:
            x = self.final_layer_down(x, edge_index, edge_weights)

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
