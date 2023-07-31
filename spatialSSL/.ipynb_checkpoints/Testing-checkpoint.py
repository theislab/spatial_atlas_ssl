from tqdm.auto import tqdm
from sklearn.metrics import r2_score
import torch

def test(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    
    total_test_loss = 0
    test_targets_list = []
    test_outputs_list = []
    test_celltypes_list = []

    for data in tqdm(test_loader):
        with torch.no_grad():
            expression = torch.tensor(data.x[0].toarray(), dtype=torch.double).to(device)
            expression_masked = torch.tensor(data.y[0].toarray(), dtype=torch.double).to(device)
            
            outputs = model(expression.float(), data.edge_index.to(device).long())
            loss = criterion(outputs[data.mask], expression_masked.float())
        total_test_loss += loss.item() * data.num_graphs
        test_celltypes_list.append(data.cell_type_masked)
        test_targets_list.append(expression_masked.float())
        test_outputs_list.append(outputs[data.mask])

    # Measure and print test loss and R2
    test_loss = total_test_loss / len(test_loader.dataset)
    test_r2 = r2_score(torch.cat(test_targets_list).cpu().detach().numpy(),
                       torch.cat(test_outputs_list).cpu().detach().numpy())
    print(f"Test loss: {test_loss:.4f}, test r2: {test_r2:.4f}")
        # Return test loss and r2 for further use
    return test_loss, test_r2,test_targets_list,test_outputs_list,test_celltypes_list
