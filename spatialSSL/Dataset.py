from torch_geometric.data import Dataset, Data
from torch_geometric.utils import k_hop_subgraph


class EgoNetDataset(Dataset):
    def __init__(self, graphs):
        super(EgoNetDataset, self).__init__()
        self.graphs = graphs

    def len(self):
        # Return the number of nodes in the dataset.
        return sum([g.x.shape[0] for g in self.graphs])

    def get(self, idx):
        #find graph and node idx
        for graph in self.graphs:
            if idx < graph.x.shape[0]:
                break
            idx -= graph.x.shape[0]

        #get subgraph
        subset, edge_index, mapping, edge_mask =  k_hop_subgraph(node_idx=[idx], edge_index=graph.edge_index, num_hops=1, relabel_nodes=True)
        return Data(x=graph.x[subset], edge_index=edge_index)