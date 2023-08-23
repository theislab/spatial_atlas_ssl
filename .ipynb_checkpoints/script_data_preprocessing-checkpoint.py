import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import FuncFormatter
import zipfile
from spatialSSL.Dataloader import FullImageDatasetConstructor

import numpy as np
from sklearn.model_selection import train_test_split

import configparser

# Load the configuration file
config = configparser.ConfigParser()
config.read('hyperparameters.ini')

# Extract paths
# PATHS
DATA_PATH = config.get('paths', 'data_path')
PREPROCESSED_IMG_PATH = config.get('paths', 'preprocessed_img_path')
PREPROCESSED_DATA_PATH = config.get('paths', 'preprocessed_data_path')

PREPROCESSED_DATA_ZIP_NAME = config.get('paths', 'preprocessed_data_zip_name')
#TUNE_DATA_ZIP_NAME = config.get('paths', 'tune_data_zip_name')

PRE_TRAIN_NCELL_IMG_NAME = config.get('paths', 'pre_train_ncell_img_name')
PRE_TRAIN_CTYPE_IMG_NAME = config.get('paths', 'pre_train_ctype_img_name')

PRE_VAL_NCELL_IMG_NAME = config.get('paths', 'pre_val_ncell_img_name')
PRE_VAL_CTYPE_IMG_NAME = config.get('paths', 'pre_val_ctype_img_name')

PRE_MODEL_STORE_PATH = config.get('paths', 'pre_model_store_path')
TRANSFER_MODEL_STORE_PATH = config.get('paths', 'transfer_model_store_path')

# Extract hyperparameters
image_col = config.get('graph_construct', 'image_col')
label_col = config.get('graph_construct', 'label_col')
radius = config.getint('graph_construct', 'radius')
node_level = config.getint('graph_construct', 'node_level')
mask_method = config.get('graph_construct', 'mask_method')
random_mask_percentage = config.getfloat('graph_construct', 'random_mask_percentage')
niche_to_mask = config.getint('graph_construct', 'niche_to_mask')
celltype_to_mask = config.get('graph_construct', 'celltype_to_mask')
include_label = config.get("graph_construct","include_label")
# encode book
category_encoding = {
    'Astro-Epen': 0,
    'CB GABA': 1,
    'CB Glut': 2,
    'CGE GABA': 3,
    'CNU GABA': 4,
    'CNU-HYa GABA': 5,
    'CNU-HYa Glut': 6,
    'HY GABA': 7,
    'HY Glut': 8,
    'HY Gnrh1 Glut': 9,
    'HY MM Glut': 10,
    'IT-ET Glut': 11,
    'Immune': 12,
    'LQ': 13,
    'LSX GABA': 14,
    'MB Dopa': 15,
    'MB GABA': 16,
    'MB Glut': 17,
    'MB-HB Sero': 18,
    'MGE GABA': 19,
    'MH-LH Glut': 20,
    'MOB-CR Glut': 21,
    'MOB-DG-IMN': 22,
    'MY GABA': 23,
    'MY Glut': 24,
    'NP-CT-L6b Glut': 25,
    'OEG': 26,
    'Oligo': 27,
    'P GABA': 28,
    'P Glut': 29,
    'Pineal Glut': 30,
    'TH Glut': 31,
    'Vascular': 32
}
import torch


def convert_to_float32(data_list):
    for data in data_list:
        # Convert attributes to float32
        data.x = data.x.to(dtype=torch.float32)
    return data_list


def thousands(x, pos):
    return f'{int(x)}k'


formatter = FuncFormatter(thousands)


def plot_histogram(graph_list, title, cell_num_path, cell_type_path):
    # Extracting number of cells in x
    num_cells = [data.x.size(0) / 1000 for data in graph_list]  # Divide by 1000 to represent in k

    # Plotting histogram for number of cells in x
    plt.figure(figsize=(10, 5))
    plt.hist(num_cells, bins=30, edgecolor='black')
    plt.title(f'Number of Cells in Images for {title}')
    plt.xlabel('Number of Cells')
    plt.ylabel('Number of Images')
    plt.yticks(np.arange(0, max(plt.yticks()[0]), 1))  # Set y-ticks to integer values
    plt.gca().xaxis.set_major_formatter(formatter)  # Apply the formatter
    plt.savefig(cell_num_path)  # Save the plot
    plt.show()

    # Extracting cell types
    cell_types = [data.cell_type.tolist() for data in graph_list]
    flat_cell_types = [item for sublist in cell_types for item in sublist]

    # Get the encode_book from the first data object in the graph_list
    encode_book = graph_list[0].encode_book

    # Count the frequency of each cell type
    cell_type_counts = Counter(flat_cell_types)

    # Sort cell types by frequency
    sorted_cell_types = sorted(cell_type_counts.items(), key=lambda x: x[1] / 1000,
                               reverse=True)  # Divide by 1000 to represent in k
    sorted_labels = [list(encode_book.keys())[cell_type[0]] for cell_type in sorted_cell_types]  # Convert keys to list
    sorted_values = [cell_type[1] for cell_type in sorted_cell_types]

    # Plotting histogram for distribution of cell types
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(sorted_values)), sorted_values, edgecolor='black')
    plt.title(f'Distribution of Cell Types for {title}')
    plt.xlabel('Cell Type')
    plt.ylabel('Number of Cells')
    plt.gca().yaxis.set_major_formatter(formatter)  # Apply the formatter
    plt.xticks(ticks=range(len(sorted_labels)), labels=sorted_labels, rotation=90)
    plt.savefig(cell_type_path)  # Save the plot
    plt.show()





# Create an instance of Full_image_dataloader

data_constracter = FullImageDatasetConstructor(file_path=DATA_PATH,
                                               image_col=image_col,
                                               label_col=label_col,
                                               radius=radius,
                                               include_label = include_label,
                                               node_level=node_level,
                                               mask_method=mask_method,
                                               random_mask_percentage=random_mask_percentage,
                                               encode_book=category_encoding,
                                               niche_to_mask=niche_to_mask,
                                               celltype_to_mask=celltype_to_mask,
                                               )
# Load the data
data_constracter.load_data()
# Construct the graph
graph_list = data_constracter.construct_graph()
graph_list = convert_to_float32(graph_list)

# Split the graph_list into 80% for pre-training and 20% for pre-training validation
pre_train_list, pre_val_list = train_test_split(graph_list, test_size=0.20, random_state=42)

import os

# Save train_loader and val_loader to pickle files
torch.save(pre_train_list, PREPROCESSED_DATA_PATH + "pre_train_list.pt")
torch.save(pre_val_list, PREPROCESSED_DATA_PATH + 'pre_val_list.pt')

# Create a ZIP file and add the pickle files to it
with zipfile.ZipFile(PREPROCESSED_DATA_PATH + PREPROCESSED_DATA_ZIP_NAME, 'w') as zipf:
    zipf.write(PREPROCESSED_DATA_PATH + 'pre_train_list.pt', arcname='pre_train_list.pt')
    zipf.write(PREPROCESSED_DATA_PATH + 'pre_val_list.pt', arcname='pre_val_list.pt')

# Delete the torch files
os.remove(PREPROCESSED_DATA_PATH + 'pre_train_list.pt')
os.remove(PREPROCESSED_DATA_PATH + 'pre_val_list.pt')

print("ZIP file created successfully!")

# Split the pre_val_list into 80% for tune_train and 20% for temporary validation/test
#tune_train, temp_val_test = train_test_split(pre_train_list, test_size=0.20, random_state=42)

# Split the temporary validation/test into 50% for tune_val and 50% for tune_test
#tune_val, tune_test = train_test_split(temp_val_test, test_size=0.50, random_state=42)

# save the tune_train, tune_val and tune_test to pt files
#torch.save(tune_train, PREPROCESSED_DATA_PATH + "tune_train.pt")
#torch.save(tune_val, PREPROCESSED_DATA_PATH + 'tune_val.pt')
#torch.save(tune_test, PREPROCESSED_DATA_PATH + 'tune_test.pt')

# Create a ZIP file and add the pickle files to it
#with zipfile.ZipFile(PREPROCESSED_DATA_PATH + PREPROCESSED_DATA_ZIP_NAME, 'w') as zipf:
#    zipf.write(PREPROCESSED_DATA_PATH + 'tune_train.pt', arcname='tune_train.pt')
#    zipf.write(PREPROCESSED_DATA_PATH + 'tune_val.pt', arcname='tune_val.pt')
#    zipf.write(PREPROCESSED_DATA_PATH + 'tune_test.pt', arcname='tune_test.pt')

# Delete the torch files
#os.remove(PREPROCESSED_DATA_PATH + 'tune_train.pt')
#os.remove(PREPROCESSED_DATA_PATH + 'tune_val.pt')
#os.remove(PREPROCESSED_DATA_PATH + 'tune_test.pt')
