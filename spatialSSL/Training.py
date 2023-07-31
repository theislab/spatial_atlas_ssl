import torch
from torch import nn, optim
from sklearn.metrics import r2_score
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, training=True, gene_expression=None):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    targets_list = []
    outputs_list = []

    with torch.set_grad_enabled(training):
        for data in loader:

            if training:
                optimizer.zero_grad()

            if gene_expression is None:
                data = data.to(device)
                outputs = model(data.x.float(), data.edge_index.long())
                loss = criterion(outputs[data.mask], data.y.float())
                targets_list.append(data.y.cpu().detach())
            else:
                input = torch.tensor(gene_expression[data.x].toarray(), dtype=torch.double).to(device)
                input[data.mask] = 0
                target = torch.tensor(gene_expression[data.y].toarray(), dtype=torch.double).to(device)
                outputs = model(input.float(), data.edge_index.to(device).long())
                loss = criterion(outputs[data.mask], target.float())
                targets_list.append(target.cpu())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_graphs
            outputs_list.append(outputs[data.mask].cpu().detach())

    return total_loss / len(loader.dataset), r2_score(torch.cat(targets_list).numpy(), torch.cat(outputs_list).numpy())


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=5, model_path=None,
          gene_expression=None):
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, criterion, gene_expression, training=True)
        val_loss, val_r2 = train_epoch(model, val_loader, criterion, gene_expression, training=False)
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
            f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}, Time: {time.time() - start_time}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")
