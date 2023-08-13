import time

import pandas as pd
import torch
from torcheval.metrics import R2Score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, gene_expression=None, training=True, output=False,
                masking_ratio=0.3):
    model.train(training)
    total_loss = 0

    r2 = R2Score().to(device)

    with torch.set_grad_enabled(training):
        for data in loader:

            if training:
                optimizer.zero_grad()

            if gene_expression is None:
                data = data.to(device)
                outputs = model(data.x.float(), data.edge_index.long())
                loss = criterion(outputs[data.mask], data.y.float())
                target = data.y
            else:

                input_model = torch.tensor(gene_expression[data.x].toarray(), dtype=torch.double).to(device)

                if True:
                    mask = torch.rand(input_model.shape)  # uniformly distributed between 0 and 1
                    mask = mask < masking_ratio
                    target = torch.tensor(gene_expression[data.x].toarray()).to(device)
                else:
                    mask = data.mask
                    target = torch.tensor(gene_expression[data.y].toarray(), dtype=torch.double).to(device)

                input_model[mask] = 0
                outputs = model(input_model.float(), data.edge_index.to(device).long(),
                                data.edge_weights.to(device).float())
                loss = criterion(outputs, target.float())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_graphs
            r2.update(outputs.flatten(), target.flatten())

            # r2.update(outputs[data.mask].flatten(), target.flatten())

    return total_loss / len(loader.dataset), r2.compute().cpu().numpy(), (
        outputs, target.float()) if output else None  # outputs[data.mask]


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=5, model_path=None,
          gene_expression=None, masking_ratio=0.3):
    model = model.to(device)

    # records losses
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    # records best val loss
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    # records time
    start_time = time.time()

    # training loop
    for epoch in tqdm(range(num_epochs), desc='Training model'):
        epoch_start_time = time.time()
        train_loss, train_r2, _ = train_epoch(model, train_loader, optimizer, criterion=criterion,
                                              gene_expression=gene_expression, training=True,
                                              masking_ratio=masking_ratio)
        val_loss, val_r2, _ = train_epoch(model, val_loader, optimizer=None, criterion=criterion,
                                          gene_expression=gene_expression, training=False, masking_ratio=masking_ratio)

        # records losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)

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
            f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")
    return TrainResults(train_losses, train_r2_scores, val_losses, val_r2_scores, best_epoch, epoch + 1,
                        time.time() - start_time, "R2_Score")


class TrainResults:
    def __init__(self, train_losses, train_r2s, val_losses, val_r2s, best_epoch, epochs_trained, total_training_time, metric_name):
        self.metric_name = metric_name
        self.train_losses = train_losses
        self.train_r2s = train_r2s
        self.val_losses = val_losses
        self.val_r2s = val_r2s
        self.best_epoch = best_epoch
        self.epochs_trained = epochs_trained
        self.total_training_time = total_training_time
        self.masking_ratio = None
        self.radius = None

    def __str__(self):
        return f"Best val loss: {self.val_losses[self.best_epoch]:.4f}, best r2 val: {self.val_r2s[self.best_epoch]:.4f}, at epoch {self.best_epoch}"

    def plot(self):

        # TODO: add vertical line at best epoch

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))  # Create a subplot with 2 rows

        # Plot the first set of data
        axes[0].plot(self.train_losses, label='Training Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].title.set_text(
            "Losses, Best Validation Loss: {:.4f}, at Epoch {}".format(self.val_losses[self.best_epoch], self.best_epoch))

        # add vertical line at best epoch, label it
        axes[0].axvline(x=self.best_epoch, color='r', linestyle='--', label='Best Epoch')
        #axes[0].text(self.best_epoch, 0.5, 'Best Epoch', rotation=90)

        axes[0].legend()

        # Plot the second set of data
        axes[1].plot(self.train_r2s, label=f'Training {self.metric_name}')
        axes[1].plot(self.val_r2s, label=f'Validation {self.metric_name}')
        axes[1].title.set_text(
            f"{self.metric_name}, Best Validation {self.metric_name}: {self.val_r2s[self.best_epoch]:.4f}, at epoch {self.best_epoch}")  #.format(self.val_r2s[self.best_epoch], self.best_epoch + 1))

        # add vertical line at best epoch, label it
        axes[1].axvline(x=self.best_epoch, color='r', linestyle='--', label='Best Epoch')
        #axes[1].text(self.best_epoch, 0.5, 'Best Epoch', rotation=90)

        axes[1].legend()

        # Return the figure object containing the subplots
        return fig


    def to_pandas(self):
        return pd.DataFrame({'best_epoch': self.best_epoch + 1, 'epochs_trained': self.epochs_trained,
                             'total_training_time': self.total_training_time,
                             'best_val_loss': self.val_losses[self.best_epoch],
                             f'best_val_{self.metric_name}': self.val_r2s[self.best_epoch]}, index=[0])
