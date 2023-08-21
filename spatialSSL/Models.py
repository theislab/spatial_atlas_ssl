from torch import nn, optim, Tensor
import torch
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import r2_score
from torch.nn import LeakyReLU, Dropout
import time
from torch.utils.checkpoint import checkpoint


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.dropout = Dropout(dropout_rate)
        
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = self.dropout(self.act(self.conv2(x, edge_index)))
        x = self.dropout(self.act(self.conv3(x, edge_index)))
        x = self.act(self.conv4(x, edge_index))  # Typically, dropout is not applied to the final layer.
        return x
    
class GCN_1(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)

        
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.act(self.conv1(x, edge_index))  # Typically, dropout is not applied to the final layer.
        return x
    
class GCN_2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = Dropout(dropout_rate)
        
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = checkpoint(self.conv2, x, edge_index)
        x = self.act(self.conv3(x, edge_index))  # Typically, dropout is not applied to the final layer.
        return x

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        self.dropout = Dropout(dropout_rate)
        
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = self.dropout(self.act(self.conv2(x, edge_index)))
        x = self.dropout(self.act(self.conv3(x, edge_index)))
        x = self.act(self.conv4(x, edge_index))  # Typically, dropout is not applied to the final layer.
        return x
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)
        self.conv4 = GATConv(hidden_channels, out_channels)
        self.dropout = Dropout(dropout_rate)
        self.act = nn.LeakyReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = self.dropout(self.act(self.conv2(x, edge_index)))
        x = self.dropout(self.act(self.conv3(x, edge_index)))
        x = self.act(self.conv4(x, edge_index))  # Typically, dropout is not applied to the final layer.
        return x
    
class GAT_2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)
        self.dropout = Dropout(dropout_rate)
        self.act = nn.LeakyReLU()
        

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.dropout(self.act(self.conv1(x, edge_index)))
        x = self.act(self.conv2( x, edge_index)) # Using checkpointing on the final layer
        return x
    
class GAT4(nn.Module):
    def __init__(self, in_channels, hidden_channels_1,hidden_channels_2, out_channels, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels_1)
        self.conv2 = GATConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GATConv(hidden_channels_2, hidden_channels_1)
        self.conv4 = GATConv(hidden_channels_1, out_channels)

        
        self.dropout = Dropout(dropout_rate)
        self.act = nn.LeakyReLU()
        
    
    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.act(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        return x

class TransformerModel(nn.Module):
    def __init__(self, in_channels, heads=2):
        super(TransformerModel, self).__init__()
        self.heads = heads
        self.transformer_conv1 = TransformerConv(in_channels, 33, heads=heads) # Output channels: 256 * heads
        self.transformer_conv2 = TransformerConv(33 * heads, 550, heads=heads) # Input channels: 256 * heads, Output channels: 550

    def forward(self, x, edge_index):
        x = self.transformer_conv1(x, edge_index)
        x = self.transformer_conv2(x, edge_index)
        # Reshape x to have the heads as a separate dimension
        x = x.view(x.size(0), 550, self.heads)

        # Apply max pooling across the heads dimension
        x, _ = torch.max(x, dim=2)
        return x


# Model for classification task, input would require the expected number of classes
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class GATClassification(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_heads=4, dropout_rate=0.2):
        super(GATClassification, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.classifier = nn.Linear(hidden_channels * num_heads, num_classes)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.classifier(x)
        return x