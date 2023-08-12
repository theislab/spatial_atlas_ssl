import time

import pandas as pd
import torch
from torcheval.metrics import R2Score, MulticlassAccuracy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from spatialSSL.Pretraining import TrainResults

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, gene_expression=None, training=True, output=False):
    model.train(training)
    total_loss = 0

    acc = MulticlassAccuracy().to(device)

    with torch.set_grad_enabled(training):
        for data in loader:

            if training:
                optimizer.zero_grad()

            input = torch.tensor(gene_expression.X[data.x].toarray(), dtype=torch.double).to(device).float()
            labels = torch.tensor(gene_expression[data.x.numpy()].obs.class_label.cat.codes.values).to(
                device).long()

            # Forward + backward + optimize
            outputs = model(input, data.edge_index)
            loss = criterion(outputs, labels)

            # Print statistics
            acc.update(outputs.argmax(dim=1), labels)
            total_loss += loss.item()

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_graphs
            acc.update(outputs.argmax(dim=1), labels)

    return total_loss / len(loader.dataset), acc.compute().cpu().numpy(), (
        outputs.argmax(dim=1), labels) if output else None


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=5, model_path=None,
          gene_expression=None):
    model = model.to(device)

    # records losses
    train_losses = []
    val_losses = []
    train_acc_scores = []
    val_acc_scores = []

    # records best val loss
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    # records time
    start_time = time.time()

    # training loop
    for epoch in tqdm(range(num_epochs), desc='Training model'):
        epoch_start_time = time.time()
        train_loss, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion=criterion,
                                              gene_expression=gene_expression, training=True)
        val_loss, val_acc, _ = train_epoch(model, val_loader, optimizer=None, criterion=criterion,
                                          gene_expression=gene_expression, training=False)

        # records losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc_scores.append(train_acc)
        val_acc_scores.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping triggered!, no improvement in val loss for {} epochs'.format(patience))
                break

        print(
            f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f},  val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")
    return TrainResults(train_losses, train_acc_scores, val_losses, val_acc_scores, best_epoch, epoch + 1,
                        time.time() - start_time, "Accuracy")
