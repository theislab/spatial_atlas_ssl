import numpy as np
import torch
from torch import nn, optim
import time
from torcheval.metrics import MeanSquaredError, R2Score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # "cpu"


def train_epoch(model, loader, optimizer, criterion, r2_metric, mse_metric, gene_expression=None, training=True,
                weight_loss=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    with torch.set_grad_enabled(training):
        for data in loader:

            if training:
                optimizer.zero_grad()
            if not training and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if gene_expression is None:

                cell_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
                cell_mask[data.cell_mask_index] = True

                target = data.x.float().to_dense()[cell_mask]
                # print(data.x.float().to_dense().shape)

                input = data.x.float().to_dense() * (~cell_mask).view(-1, 1)
                # print(input.shape)

                outputs = model(input.to(device), data.edge_index.long().to(device))
                # print(outputs)

                if weight_loss:
                    feature_weights = get_feature_weights(
                        data.cell_type[data.cell_mask_index])  # Get the weights for this image

                    squared_errors = (outputs[cell_mask] - target.to(device)) ** 2

                    # Make sure the weights have the same shape as the masked outputs
                    feature_weights = feature_weights.view(-1, 1).expand_as(squared_errors)

                    # Multiply the squared errors by the weights
                    weighted_errors = squared_errors * feature_weights.to(device)

                    loss = weighted_errors.mean()
                else:
                    loss = criterion(outputs[cell_mask], target.to(device))
                # evaluate metrics

                r2_metric.update(outputs[cell_mask].detach().cpu(), target.detach().cpu())
                mse_metric.update(outputs[cell_mask].detach().cpu(), target.detach().cpu())



            else:
                input = torch.tensor(gene_expression[data.x].toarray(), dtype=torch.double).to(device)
                input[data.mask] = 0
                target = torch.tensor(gene_expression[data.y].toarray(), dtype=torch.double).to(device)
                outputs = model(input.float(), data.edge_index.to(device).long())
                loss = criterion(outputs[data.mask], target)

            if training:
                loss.backward()
                optimizer.step()

            # Clear unnecessary tensors to free up memory
            del input, target, outputs
            torch.cuda.empty_cache()

            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset), r2_metric.compute(), mse_metric.compute()


def get_feature_weights(data_cell_types):
    # Convert the tensor to a NumPy array
    data_cell_types_np = data_cell_types.cpu().numpy()

    # Count the frequency of each cell type
    unique, counts = np.unique(data_cell_types_np, return_counts=True)
    cell_type_counts = dict(zip(unique, counts))

    # Calculate the total number of cell types
    total_cell_types = sum(cell_type_counts.values())

    # Calculate the inverse frequency weights
    weights = {cell_type: total_cell_types / count for cell_type, count in cell_type_counts.items()}

    # Normalize the weights so they sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {cell_type: weight / total_weight for cell_type, weight in weights.items()}

    # Convert to a tensor
    weight_tensor = torch.tensor([normalized_weights[cell_type] for cell_type in data_cell_types_np], dtype=torch.float)

    return weight_tensor



def train(model, train_loader, val_loader, criterion, num_epochs=100, patience=5, optimizer=None, model_path=None,
          gene_expression=None, weight_loss=False):
    r2_metric_train = R2Score()
    r2_metric_val = R2Score()
    mse_metric_train = MeanSquaredError()
    mse_metric_val = MeanSquaredError()

    # model.to(device)
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_r2, train_mse = train_epoch(model, train_loader, optimizer, criterion, r2_metric_train,
                                                      mse_metric_train, gene_expression, training=True,
                                                      weight_loss=weight_loss)

        val_loss, val_r2, val_mse = train_epoch(model, val_loader, optimizer, criterion, r2_metric_val, mse_metric_val,
                                                gene_expression, weight_loss=weight_loss, training=False)
        # scheduler.step() # Decrease learning rate by scheduler

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f}, train mse: {train_mse:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}, val mse: {val_mse:.4f}, Time: {time.time() - start_time:.4f}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")
