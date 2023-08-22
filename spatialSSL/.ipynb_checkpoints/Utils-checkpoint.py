from torch.utils.data import random_split, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

def split_dataset(dataset, split_percent=(0.8, 0.1, 0.1), batch_size=64, pre_train=False, k_folds=None):
    if k_folds is not None:
        # K-Fold Cross Validation
        kf = KFold(n_splits=k_folds, shuffle=True)
        loaders = []
        for train_index, val_index in kf.split(dataset):
            train_data = [dataset[i] for i in train_index]
            val_data = [dataset[i] for i in val_index]
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
            loaders.append((train_loader, val_loader))
        return loaders
    else:
        if not pre_train:
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
        if pre_train:
                    # Calculate split sizes
            total_len = len(dataset)
            train_len = int(total_len * split_percent[0])
            val_len = total_len - train_len
            train_data, val_data = random_split(dataset, [train_len, val_len])
            # Create data loaders for each set
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            return train_loader, val_loader

class graphDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    
def visualize_cell_type_accuracies(cell_type_accuracies):
    plt.bar(range(len(cell_type_accuracies)), cell_type_accuracies)
    plt.xlabel('Cell Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Each Cell Type')
    plt.show()