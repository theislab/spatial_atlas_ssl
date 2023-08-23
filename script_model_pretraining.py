import configparser

import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FuncFormatter
import zipfile
from spatialSSL.Models import *
from spatialSSL.Dataloader import FullImageDatasetConstructor
from spatialSSL.Utils import split_dataset,visualize_cell_type_accuracies
from spatialSSL.Training import train,train_classification,test_classification
from spatialSSL.Models import GAT4,Transferlearn
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Define a function to load the data from the ZIP file
def load_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(file_name) as file:
            return torch.load(file)

# Load the configuration file
config = configparser.ConfigParser()
config.read('hyperparameters.ini')

in_channels = config.getint('pre_train_hyperparam', 'in_channels')
hidden_channels_1 = config.getint('pre_train_hyperparam', 'hidden_channels_1')
hidden_channels_2 = config.getint('pre_train_hyperparam', 'hidden_channels_2')
out_channels = config.getint('pre_train_hyperparam', 'out_channels')
dropout = config.getfloat('pre_train_hyperparam', 'dropout')
lr = config.getfloat('pre_train_hyperparam', 'lr')
num_epochs = config.getint('pre_train_hyperparam', 'num_epochs')
patience = config.getint('pre_train_hyperparam', 'patience')

# boolean
weight_loss = config.getboolean('pre_train_hyperparam', 'weight_loss')

# Extract paths
PREPROCESSED_DATA_PATH = config.get('paths', 'preprocessed_data_path')
PREPROCESSED_DATA_ZIP_NAME = config.get('paths', 'preprocessed_data_zip_name')
PRE_MODEL_STORE_PATH = config.get('paths', 'pre_model_store_path')

# Load the pre_train_list and pre_val_list from the ZIP file
pre_train_list = load_from_zip(PREPROCESSED_DATA_PATH+PREPROCESSED_DATA_ZIP_NAME, 'pre_train_list.pt')
pre_val_list = load_from_zip(PREPROCESSED_DATA_PATH+PREPROCESSED_DATA_ZIP_NAME, 'pre_val_list.pt')

# Create DataLoader objects for pre-training and pre-validation
pre_train_loader = DataLoader(pre_train_list, batch_size=1, shuffle=True)
pre_val_loader = DataLoader(pre_val_list, batch_size=1, shuffle=False)


# Run pretraining



# Pretraining
# Define the device
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') #"cpu"

# Create the model
model = GAT4(in_channels, hidden_channels_1,hidden_channels_2, out_channels).to(device) # in_channels is set to 100 as an example. Please replace it with your actual feature size.

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)



train(model=model, train_loader=pre_train_loader, val_loader=pre_val_loader, criterion=criterion, num_epochs= num_epochs, patience = patience, optimizer= optimizer,weight_loss = weight_loss, model_path=PRE_MODEL_STORE_PATH)