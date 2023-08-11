import os
import sys

import matplotlib.pyplot as plt
import torch
import scanpy as sc
from torch_geometric.loader import DataLoader
from matplotlib.backends.backend_pdf import PdfPages


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL


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
adata = sc.read(args[9])

obs_df = adata.obs.copy()
sorted_obs_df = obs_df.sort_values(by="section")
adata = adata[sorted_obs_df.index]

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

model = spatialSSL.Models.GCN(in_channels=550, hidden_channels=550, out_channels=550)


criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


training_summary = spatialSSL.Pretraining.train(model, train_loader, val_loader, optimizer, criterion, num_epochs, patience, model_path = output_model, gene_expression=adata.X, masking_ratio=masking_mode)


# Save training summary to pdf
with PdfPages(args[10]) as pdf:
    fig = training_summary.plot()
    pdf.savefig(fig)

# save training summary to csv
training_summary.to_pandas().to_csv(args[11], index=False)