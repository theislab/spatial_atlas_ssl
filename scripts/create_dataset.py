import sys
import os
import torch
from torch.utils.data import random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

args = sys.argv

file_path = args[1]
# "/data/ceph/hdd/project/node_05/gene2bird/groupB/genomic/atlas_brain_638850_CCF.h5ad"

# Create the dataloader
dataset_constructor = spatialSSL.Dataloader.EgoNetDatasetConstructor(file_path=file_path, image_col="section",
                                                                     label_col="class_label", include_label=False,
                                                                     radius=int(args[2]), node_level=int(args[3]))

# Load the data
dataset_constructor.load_data()

# Construct
dataset = dataset_constructor.construct_graph(show_progress_bar=True)

# Split the dataset into train, validation, and test sets
train_data, val_data, test_data = random_split(dataset, (0.8,0.1,0.1))

# Save the data
torch.save(train_data, args[4] + "train.pt")
torch.save(val_data, args[4] + "val.pt")
torch.save(test_data, args[4] + "test.pt")

# "/data/ceph/hdd/project/node_05/gene2bird/groupB/spatial_atlas_ssl/datasets/full_dataset_radius_20_khop_3.pt")
