import configparser

import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FuncFormatter
import zipfile
from spatialSSL.Models import *
from spatialSSL.Dataloader import FullImageDatasetConstructor
from spatialSSL.Training import train,train_classification,test_classification
from spatialSSL.Models import GAT4,Transferlearn
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from spatialSSL.Utils import split_dataset,visualize_cell_type_accuracies
# Load the configuration file
config = configparser.ConfigParser()
config.read('hyperparameters.ini')

in_channels = config.getint('pre_train_hyperparam', 'in_channels')
hidden_channels_1 = config.getint('pre_train_hyperparam', 'hidden_channels_1')
hidden_channels_2 = config.getint('pre_train_hyperparam', 'hidden_channels_2')
out_channels = config.getint('pre_train_hyperparam', 'out_channels')
dropout = config.getfloat('pre_train_hyperparam', 'dropout')
num_epochs = config.getint('pre_train_hyperparam', 'num_epochs')
patience = config.getint('pre_train_hyperparam', 'patience')
num_classes = config.getint('pre_train_hyperparam', 'num_classes')
lr_transfer = config.getfloat('pre_train_hyperparam', 'lr_transfer')

# boolean
freeze = config.getboolean('pre_train_hyperparam', 'freeze')

# paths
# Extract paths
PREPROCESSED_DATA_PATH = config.get('paths', 'preprocessed_data_path')
PREPROCESSED_DATA_ZIP_NAME = config.get('paths', 'preprocessed_data_zip_name')
PRE_MODEL_STORE_PATH = config.get('paths', 'pre_model_store_path')

# Load the pre_train_list and pre_val_list from the ZIP file
def load_from_zip(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(file_name) as file:
            return torch.load(file)

pre_val_list = load_from_zip(PREPROCESSED_DATA_PATH+PREPROCESSED_DATA_ZIP_NAME, 'pre_val_list.pt')

tune_train, temp_val_test = train_test_split(pre_val_list, test_size=0.20, random_state=42)

# Split the temporary validation/test into 50% for tune_val and 50% for tune_test
tune_val, tune_test = train_test_split(temp_val_test, test_size=0.50, random_state=42)

# Create DataLoader objects for pre-training and pre-validation
tune_train_loader = DataLoader(tune_train, batch_size=1, shuffle=True)
tune_val_loader = DataLoader(tune_val, batch_size=1, shuffle=False)
tune_test_loader = DataLoader(tune_test, batch_size=1, shuffle=False)

# transfer learning

device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu') #"cpu"
pretrained_model = GAT4(in_channels, hidden_channels_1,hidden_channels_2, out_channels)

if freeze:
    for param in pretrained_model.parameters():
        param.requires_grad = False

model = Transferlearn(pretrained_model,out_channels,hidden_channels_1,num_classes).to(device)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr_transfer)

train_classification(model=model, train_loader=tune_train_loader, val_loader=tune_val_loader, num_epochs=num_epochs_transfer,criterion=criterion, num_classes=num_classes,optimizer=optimizer,model_path=TRANSFER_MODEL_STORE_PATH)

cell_type_accuracies = test_classification(model, tune_test_loader, criterion, num_classes)
visualize_cell_type_accuracies(cell_type_accuracies)