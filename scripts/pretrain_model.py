import os
import sys

import matplotlib.pyplot as plt
import torch
import scanpy as sc
from torch_geometric.loader import DataLoader
from matplotlib.backends.backend_pdf import PdfPages


#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

torch.manual_seed(0)

# ['scripts/pretrain_model.py', 'snake_output/dataset_3_20_train.pt', 'snake_output/dataset_3_20_val.pt', 'snake_output/model_3_20_0.5.pt', '0.5', '10', '64', '0.001']
args = sys.argv

trainset = torch.load(args[1])
valset = torch.load(args[2])
output_model = args[3]
masking_mode = float(args[4])
patience = int(args[5])
batch_size = int(args[6])
learning_rate = float(args[7])
num_epochs = int(args[8])
bottle_neck = int(args[9])
adata = sc.read(args[10])
pdf_output = args[11]
csv_output = args[12]
model_type = args[13]
num_hidden_channels = int(args[14])

# sort adata by section, we always do this to ensure that the order of the sections is the same as in the dataset
obs_df = adata.obs.copy()
sorted_obs_df = obs_df.sort_values(by="section")
adata = adata[sorted_obs_df.index]

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)


if model_type == "GCN":
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=bottle_neck, num_hidden_layers=num_hidden_channels, type="GCN")
elif model_type == "GAT":
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=bottle_neck, num_hidden_layers=num_hidden_channels, type="GAT")
else:
    raise ValueError("model_type must be either GCN or GAT")


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


training_summary = spatialSSL.Pretraining.train(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience, model_path = output_model, gene_expression=adata.X, masking_ratio=masking_mode)


# Save training summary to pdf
with PdfPages(pdf_output) as pdf:
    fig = training_summary.plot()
    pdf.savefig(fig)

# save training summary to csv
training_summary.to_pandas().to_csv(csv_output, index=False)