import numpy as np
import scanpy as sc
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
import glob
import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spatialSSL

#         python scripts/eval_models.py {params.folder} {output} {input.adata} {input.testset}

args = sys.argv


# read adata and sort adata by section
input_folder = args[1]
adata = sc.read(args[3])
obs_df = adata.obs.copy()
sorted_obs_df = obs_df.sort_values(by="section")
adata = adata[sorted_obs_df.index]

# add class_id column with integer values
category_to_int = {category: i for i, category in enumerate(adata.obs.class_label.unique())}
cat_values = adata.obs["class_label"].map(category_to_int)
adata.obs["class_id"] = cat_values

# create test loader
testset = torch.load(args[4])
testloader = DataLoader(testset, batch_size=128, shuffle=False)



# get all files from folder which end on .csv
matching_files = glob.glob(input_folder + "summary_*.csv")



# read all files and concatenate them pandas, add file basename as column
df = pd.concat([pd.read_csv(f).assign(file=os.path.basename(f)) for f in matching_files])

accs = []
bal_accs = []
f1_micros = []
f1_macros = []
f1_weighteds = []
precisions = []
recalls = []

for row in df.iterrows():
    print(row[1]['file'])

    weights = torch.load(
        input_folder +
        row[1]['model_name'])
    info = pd.read_csv(
        input_folder +
        row[1]['file'])
    info_dict = info.to_dict('records')[0]
    model = spatialSSL.PretrainModels.GCN_1(in_channels=adata.n_vars, hidden_channels=info_dict['hidden_channels'],
                                            num_classes=len(adata.obs.class_label.unique()),
                                            type=info_dict['model_type'],
                                            num_hidden_layers=info_dict['num_hidden_layers'])

    # load pretrained weights of first layer and hidden layers
    print(model.load_state_dict(weights))

    model.eval()

    gene_expression = adata
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = []
    trues = []
    with torch.set_grad_enabled(False):
        for data in testloader:
            input = torch.tensor(gene_expression.X[data.x].toarray(), dtype=torch.double).to(device).float()
            labels = torch.tensor(gene_expression[data.x.numpy()].obs['class_id']).to(
                device).long()
            outputs = model(input, data.edge_index.to(device).long(), data.edge_weights.to(device).float())
            preds.append(outputs.argmax(dim=1).detach().numpy())
            trues.append(labels.detach().numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    accs.append(accuracy_score(trues, preds))
    bal_accs.append(balanced_accuracy_score(trues, preds))
    f1_micros.append(f1_score(trues, preds, average='micro'))
    f1_macros.append(f1_score(trues, preds, average='macro'))
    f1_weighteds.append(f1_score(trues, preds, average='weighted'))
    precisions.append(precision_score(trues, preds, average='micro'))
    recalls.append(recall_score(trues, preds, average='micro'))

df['acc'] = accs
df['bal_acc'] = bal_accs
df['f1_micro'] = f1_micros
df['f1_macro'] = f1_macros
df['f1_weighted'] = f1_weighteds
df['precision'] = precisions
df['recall'] = recalls

df.to_csv(args[2], index=False)
