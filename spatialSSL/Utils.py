from torch.utils.data import random_split, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import NeighborSampler
import torch
def split_dataset(dataset, split_percent=(0.8, 0.1, 0.1), batch_size=64):
    # Split the dataset into train, validation, and test sets
    # dataset.shuffle()

    # Calculate split sizes
    total_len = len(dataset)
    train_len = int(total_len * split_percent[0])
    val_len = int(total_len * split_percent[1])
    test_len = total_len - train_len - val_len

    dataset = graphDataset(dataset)

    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

    # Create data loaders for each set
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def split_dataset_minibatch(graph_list, split_percent=(0.8, 0.1, 0.1), batch_size=1):

    # Compute the lengths of splits
    total_length = len(graph_list)
    lengths = [int(total_length * split) for split in split_percent]
    lengths[-1] = total_length - sum(lengths[:-1])  # So that sum(lengths) == total_length

    # Generate a random permutation
    indices = torch.randperm(total_length)

    # Split the graph_list according to lengths
    train_data, val_data, test_data = [graph_list[indices[i - length:i]] for i, length in zip(torch._utils._accumulate(lengths), lengths)]

    # Create NeighborSampler for each set
    train_loader = NeighborSampler(data=train_data, size=[-1]*2, batch_size=batch_size, shuffle=True)
    val_loader = NeighborSampler(data=val_data, size=[-1]*2, batch_size=batch_size, shuffle=False)
    test_loader = NeighborSampler(data=test_data, size=[-1]*2, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class graphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
