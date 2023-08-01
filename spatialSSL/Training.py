import torch
from torch import nn, optim
from sklearn.metrics import r2_score as sk_r2_score
import time
from torcheval.metrics import R2Score
from torcheval.metrics.functional import r2_score

from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, gene_expression=None, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    targets_list = []
    outputs_list = []


    r2 = R2Score()


    with torch.set_grad_enabled(training):
        for data in loader:



            if training:
                optimizer.zero_grad()

            if gene_expression is None:
                data = data.to(device)
                outputs = model(data.x.float(), data.edge_index.long())
                loss = criterion(outputs[data.mask], data.y.float())
                target = data.y
                #targets_list.append(data.y.cpu().detach())
            else:
                input = torch.tensor(gene_expression[data.x].toarray(), dtype=torch.double).to(device)
                input[data.mask] = 0
                target = torch.tensor(gene_expression[data.y].toarray(), dtype=torch.double).to(device)
                outputs = model(input.float(), data.edge_index.to(device).long())
                loss = criterion(outputs[data.mask], target.float())
                #targets_list.append(target.cpu())

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.num_graphs
            r2.update(outputs[data.mask].flatten(), target.flatten())
            #outputs_list.append(outputs[data.mask].cpu().detach())

    return total_loss / len(loader.dataset), r2.compute().cpu().numpy()#r2_score(torch.cat(targets_list).numpy(), torch.cat(outputs_list).numpy())


def train(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=5, model_path=None,
          gene_expression=None):

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
        train_loss, train_r2 = train_epoch(model, train_loader, optimizer, criterion=criterion,
                                           gene_expression=gene_expression, training=True)
        val_loss, val_r2 = train_epoch(model, val_loader, optimizer=None, criterion=criterion,
                                       gene_expression=gene_expression, training=False)

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

        print(f"Epoch {epoch + 1}/{num_epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}, Time: {time.time() - epoch_start_time:.2f}s")

    print(f"Best val loss: {best_val_loss:.4f}, at epoch {best_epoch + 1}")
    return TrainResults(train_losses, train_r2_scores, val_losses, val_r2_scores, best_epoch, epoch + 1, time.time() - start_time)


class TrainResults():
    def __init__(self, train_losses, train_r2s, val_losses, val_r2s, best_epoch, epochs_trained, total_training_time):
        self.train_losses = train_losses
        self.train_r2s = train_r2s
        self.val_losses = val_losses
        self.val_r2s = val_r2s
        self.best_epoch = best_epoch
        self.epochs_trained = epochs_trained
        self.total_training_time = total_training_time

    def __str__(self):
        return f"Best val loss: {self.val_losses[self.best_epoch]:.4f}, best r2 val: {self.val_r2s[self.best_epoch]:.4f}, at epoch {self.best_epoch + 1}"

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.train_losses, label='train loss')
        plt.plot(self.val_losses, label='val loss')
        plt.legend()
        plt.show()

        plt.plot(self.train_r2s, label='train r2')
        plt.plot(self.val_r2s, label='val r2')
        plt.legend()
        plt.show()
