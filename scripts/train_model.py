import os
import sys

import matplotlib.pyplot as plt
import torch
import scanpy as sc
from torch_geometric.loader import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

# ['scripts/train_model.py', 'snake_output/datasets/dataset_2_20_train.pt', 'snake_output/datasets/dataset_2_20_val.pt', 'snake_output/models/model_2_20_0.5_GCN.pt', '10', '64', '0.001', '100', 'example_files/img_1199670929.h5ad', 'snake_output/models/summary_2_20_0.5_GCN.pdf', 'snake_output/models/summary_2_20_0.5_GCN.csv', 'snake_output/pretrain_models/model_2_20_0.5_GCN.pt']
os.chdir("/Users/leopoldendres/Documents/Bioinformatik/MasterStudium/spatial_atlas_ssl")

args = sys.argv

print(args)

trainset = torch.load(args[1])
valset = torch.load(args[2])
output_model = args[3]
patience = int(args[4])
batch_size = int(args[5])
learning_rate = float(args[6])
num_epochs = int(args[7])
adata = sc.read(args[8])
pretrained_model = args[11]

# sort adata by section
obs_df = adata.obs.copy()
sorted_obs_df = obs_df.sort_values(by="section")
adata = adata[sorted_obs_df.index]

# create train and val loaders
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# initialize model randomly
if pretrained_model == "None":
    hidden_channels = int(args[12])
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=hidden_channels,
                                                  num_classes=len(adata.obs.class_label.unique()))
else: # load pretrained model
    # load pretrained model
    pretrained_weights = torch.load(pretrained_model)

    # bottleneck layer size
    hidden_channels = list(pretrained_weights.items())[0][1].shape[0]
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=hidden_channels,
                                            num_classes=len(adata.obs.class_label.unique()))

    # load pretrained weights of first layer and hidden layers
    model.load_state_dict(pretrained_weights, strict=False)
    #model.preconv1.weight = torch.nn.Parameter(pretrained_weights['layer1.lin.weight'])
    #model.preconv1.bias = torch.nn.Parameter(pretrained_weights['layer1.bias'])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_summary = spatialSSL.Training.train(model, train_loader, val_loader, optimizer, criterion, num_epochs,
                                             patience, model_path=output_model, gene_expression=adata)

# Save training summary to pdf
with PdfPages(args[9]) as pdf:
    fig = training_summary.plot()
    pdf.savefig(fig)

# save training summary to csv
training_summary.to_pandas().to_csv(args[10], index=False)
