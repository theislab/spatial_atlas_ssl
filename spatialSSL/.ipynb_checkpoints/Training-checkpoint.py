import torch
from torch import nn, optim
from sklearn.metrics import r2_score
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    targets_list = []
    outputs_list = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x.float(), data.edge_index.long())
        loss = criterion(outputs[~data.mask], data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        targets_list.append(data.y.cpu().detach())
        outputs_list.append(outputs[~data.mask].cpu().detach())

    return total_loss / len(loader.dataset), r2_score(torch.cat(targets_list).numpy(), torch.cat(outputs_list).numpy())


def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data.x.float(), data.edge_index.long())
            loss = criterion(outputs[~data.mask], data.y.float())

            total_loss += loss.item() * data.num_graphs
            targets_list.append(data.y.cpu())
            outputs_list.append(outputs[~data.mask].cpu())

    return total_loss / len(loader.dataset), r2_score(torch.cat(targets_list).numpy(), torch.cat(outputs_list).numpy())


def train(model, train_loader, val_loader,optimizer, criterion, num_epochs, patience, model_name = 'model.pt'):
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_r2 = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_r2 = validate_one_epoch(model, val_loader, criterion)
        # scheduler.step() # Decrease learning rate by scheduler

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_name)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}, Time: {time.time() - start_time}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")