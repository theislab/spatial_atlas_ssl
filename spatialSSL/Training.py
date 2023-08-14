import torch
from torch import nn, optim
import time
from torcheval.metrics import MeanSquaredError, R2Score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # "cpu"


def train_epoch(model, loader, optimizer, criterion, r2_metric, mse_metric, gene_expression=None, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0

    with torch.set_grad_enabled(training):
        for data in loader:

            if training:
                optimizer.zero_grad()

            if gene_expression is None:

                cell_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
                cell_mask[data.cell_mask_index] = True

                target = data.x.float().to_dense()[cell_mask]
                # print(data.x.float().to_dense().shape)

                input = data.x.float().to_dense() * (~cell_mask).view(-1, 1)
                # print(input.shape)

                outputs = model(input.to(device), data.edge_index.long().to(device))
                # print(outputs)
                loss = criterion(outputs[cell_mask], target)
                # loss = (outputs[data.mask] - target).coalesce().values().pow(2).mean()
                # evaluate metrics

                r2_metric.update(outputs[cell_mask].cpu(), target.cpu())
                mse_metric.update(outputs[cell_mask].cpu(), target.cpu())

            else:
                input = torch.tensor(gene_expression[data.x].toarray(), dtype=torch.double).to(device)
                input[data.mask] = 0
                target = torch.tensor(gene_expression[data.y].toarray(), dtype=torch.double).to(device)
                outputs = model(input.float(), data.edge_index.to(device).long())
                loss = criterion(outputs[data.mask], target)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset), r2_metric.compute(), mse_metric.compute()


def train(model, train_loader, val_loader, criterion, num_epochs=100, patience=5, optimizer=None, model_path=None,
          gene_expression=None):
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
                                                      mse_metric_train, gene_expression, training=True)
        val_loss, val_r2, val_mse = train_epoch(model, val_loader, optimizer, criterion, r2_metric_val, mse_metric_val,
                                                gene_expression, training=False)
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



