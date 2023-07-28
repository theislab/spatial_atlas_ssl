import torch
from sklearn.metrics import r2_score
from torch import nn, optim


def train_model(model, train_loader, val_loader, epochs=100, lr=0.005, patience=5):
    # Set device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # store losses
    train_losses = []
    val_losses = []

    # store r2 scores
    train_r2_scores = []
    val_r2_scores = []

    best_val_loss = float('inf')
    epochs_no_improve = 0  # Number of epochs with no improvement in validation loss
    best_epoch = 0  # Epoch at which we get the best validation loss

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        targets_list = []
        outputs_list = []

        for data in train_loader:
            # Transfer data to GPU
            data = data.to(device)

            # Forward pass
            outputs = model(data.x.float(), data.edge_index.long())
            loss = criterion(outputs[~data.mask], data.y.float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Measure train loss and r2 score
            total_loss += loss.item() * data.num_graphs
            targets_list.append(data.y.float())
            outputs_list.append(outputs[~data.mask])

        # measure and print r2 and train loss
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        train_r2 = r2_score(torch.cat(targets_list).cpu().detach().numpy(),
                            torch.cat(outputs_list).cpu().detach().numpy())
        train_r2_scores.append(train_r2)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        val_targets_list = []
        val_outputs_list = []

        for data in val_loader:
            data = data.to(device)

            with torch.no_grad():
                outputs = model(data.x.float(), data.edge_index.long())
                loss = criterion(outputs[~data.mask], data.y.float())

            total_val_loss += loss.item() * data.num_graphs
            val_targets_list.append(data.y.float())
            val_outputs_list.append(outputs[~data.mask])

        # Measure and print validation loss and R2
        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_r2 = r2_score(torch.cat(val_targets_list).cpu().detach().numpy(),
                          torch.cat(val_outputs_list).cpu().detach().numpy())
        val_r2_scores.append(val_r2)

        # Early stopping and saving best parameters
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), '../models/best_model.pt')
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == patience:
                print(
                    f'Early stopping! Epoch: {epoch}, Best Epoch: {best_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                break

        print(
            f"Epoch {epoch + 1}/{epochs}, train loss: {train_loss:.4f}, train r2: {train_r2:.4f},  val loss: {val_loss:.4f}, val r2: {val_r2:.4f}")

    return train_losses, val_losses, train_r2_scores, val_r2_scores