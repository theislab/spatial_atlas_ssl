from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

def split_dataset(dataset, split_percent=(0.8, 0.1, 0.1), batch_size=64):
    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = random_split(dataset, split_percent)

    # Create data loaders for each set
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
