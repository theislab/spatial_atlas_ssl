import sys
import os
import torch
from torch.utils.data import random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

args = sys.argv

file_path = args[1]

# Create the dataloader
dataset_constructor = spatialSSL.Dataloader.EgoNetDatasetConstructor(file_path=file_path, image_col="section",
                                                                     label_col="class_label", include_label=False,
                                                                     radius=int(args[2]), node_level=int(args[3]))

# Load the data
dataset_constructor.load_data()

# Construct
dataset = dataset_constructor.construct_graph(show_progress_bar=False)

# Split the dataset into train, validation, and test sets, split test for downstream
train_data, val_data, test_data = random_split(dataset, (0.8,0.1,0.1))
# print("sizes of pre: ", len(train_data), len(val_data), len(test_data))

down_train, down_val, down_test = random_split(test_data, (0.8,0.1,0.1))
# print("sizes of down: ", len(down_train), len(down_val), len(down_test))

# Save pretrain data
torch.save(train_data, args[4] + "train.pt")
torch.save(val_data, args[4] + "val.pt")
torch.save(test_data, args[4] + "test.pt")

# Save the downstream data
# torch.save(down_train, args[5] + "train.pt")
# torch.save(down_val, args[5] + "val.pt")
# torch.save(down_test, args[5] + "test.pt")