import os
import sys

import torch
import scanpy as sc
from torch_geometric.loader import DataLoader
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

torch.manual_seed(0)

# ['scripts/train_model.py', 'snake_output/datasets/dataset_2_20_train.pt', 'snake_output/datasets/dataset_2_20_val.pt', 'snake_output/models/model_2_20_0.5_GCN.pt', '10', '64', '0.001', '100', 'example_files/img_1199670929.h5ad', 'snake_output/models/summary_2_20_0.5_GCN.pdf', 'snake_output/models/summary_2_20_0.5_GCN.csv', 'snake_output/pretrain_models/model_2_20_0.5_GCN.pt']
os.chdir("/Users/leopoldendres/Documents/Bioinformatik/MasterStudium/spatial_atlas_ssl")

args = sys.argv

# print(args)

trainset = torch.load(args[1])
valset = torch.load(args[2])
output_model = args[3]
patience = int(args[4])
batch_size = int(args[5])
learning_rate = float(args[6])
num_epochs = int(args[7])
adata = sc.read(args[8])
model_type = args[11]
pdf_output = args[9]
csv_output = args[10]
pretrained_model = args[12]

# sort adata by section
obs_df = adata.obs.copy()
sorted_obs_df = obs_df.sort_values(by="section")
adata = adata[sorted_obs_df.index]

category_to_int = {category: i for i, category in enumerate(adata.obs.class_label.unique())}
cat_values = adata.obs["class_label"].map(category_to_int)
adata.obs["class_id"] = cat_values

# create train and val loaders
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

# ['scripts/train_model.py', 'snake_output/img_1199670929.h5ad/train_datasets/dataset_2_20_train.pt', 'snake_output/img_1199670929.h5ad/train_datasets/dataset_2_20_val.pt', 'snake_output/img_1199670929.h5ad/train_models/model_2_20_0.9_pre_GAT_lay4_n64.pt', '10', '64', '0.002', '2', 'example_files/img_1199670929.h5ad', 'snake_output/img_1199670929.h5ad/train_models/summary_2_20_0.9_pre_GAT_lay4_n64.pdf', 'snake_output/img_1199670929.h5ad/train_models/summary_2_20_0.9_pre_GAT_lay4_n64.csv', 'None', 'snake_output/img_1199670929.h5ad/pretrain_models/model_2_20_0.9_pre_GAT_lay4_n64.pt']


# initialize model randomly
if pretrained_model == "None":
    hidden_channels = int(args[13])
    num_hidden_layers = int(args[14])
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=hidden_channels,
                                            num_classes=len(adata.obs.class_label.unique()), type=model_type,
                                            num_hidden_layers=num_hidden_layers)
else:  # load pretrained model
    # load pretrained model
    pretrained_weights = torch.load(pretrained_model)

    # get number of hidden channels and model type
    if "GCN" in pretrained_model:
        hidden_channels = list(pretrained_weights.items())[0][1].shape[0]
        num_hidden_layers = int(len(pretrained_weights) / 2 - 2)
        model_type = "GCN"
    elif "GAT" in pretrained_model:
        hidden_channels = list(pretrained_weights.items())[0][1].shape[2]
        num_hidden_layers = int(len(pretrained_weights) / 5 - 2)  # list(pretrained_weights.items())[0][1].shape[0]
        model_type = "GAT"
    else:
        raise ValueError("pretrained model must be either GCN or GAT")

    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=hidden_channels,
                                            num_classes=len(adata.obs.class_label.unique()), type=model_type,
                                            num_hidden_layers=num_hidden_layers)

    # load pretrained weights of first layer and hidden layers
    model.load_state_dict(pretrained_weights, strict=False)
    # model.preconv1.weight = torch.nn.Parameter(pretrained_weights['layer1.lin.weight'])
    # model.preconv1.bias = torch.nn.Parameter(pretrained_weights['layer1.bias'])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

training_summary = spatialSSL.Training.train(model, train_loader, val_loader, optimizer, criterion, num_epochs,
                                             patience, model_path=output_model, gene_expression=adata)

df = training_summary.to_pandas()

df["model_type"] = model_type
df["num_hidden_layers"] = num_hidden_layers
df["hidden_channels"] = hidden_channels
df["learning_rate"] = learning_rate
df["batch_size"] = batch_size
df["patience"] = patience
df["num_epochs"] = num_epochs
df['model_name'] = os.path.basename(output_model)
df['pretrained_model'] = True if pretrained_model != "None" else False

# Save training summary to pdf
with PdfPages(pdf_output) as pdf:
    fig = training_summary.plot()
    pdf.savefig(fig)

# save training summary to csv
df.to_csv(csv_output, index=False)
