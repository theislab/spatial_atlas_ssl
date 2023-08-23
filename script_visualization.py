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

# Plot histograms for pre_train_list and pre_val_list
plot_histogram(pre_train_list, 'Pre-Train', PREPROCESSED_IMG_PATH + PRE_TRAIN_NCELL_IMG_NAME, PREPROCESSED_IMG_PATH+ PRE_TRAIN_CTYPE_IMG_NAME)
plot_histogram(pre_val_list, 'Pre-Validation', PREPROCESSED_IMG_PATH + PRE_VAL_NCELL_IMG_NAME, PREPROCESSED_IMG_PATH+ PRE_VAL_CTYPE_IMG_NAME)


cell_type_accuracies = test_classification(model, tune_test_loader, criterion, num_classes)