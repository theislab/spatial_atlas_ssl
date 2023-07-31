import os

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import k_hop_subgraph


class EgoNetDataset(Dataset):
    def __init__(self, graphs, num_hops=1):
        super(EgoNetDataset, self).__init__()
        self.num_hops = num_hops
        self.graphs = graphs

    def len(self):
        # Return the number of nodes in the dataset.
        return sum([g.x.shape[0] for g in self.graphs])

    def get(self, idx):
        # find graph and node idx
        for graph in self.graphs:
            if idx < graph.x.shape[0]:
                break
            idx -= graph.x.shape[0]

        try:
            # calculate the subgraph
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=[idx], edge_index=graph.edge_index,
                                                                    num_hops=self.num_hops, relabel_nodes=True)
        except:
            print("Error in get function")
            print("idx: ", idx)
            print("graph.x.shape: ", graph.x.shape)
            print("graph", graph.image)

        # get subgraph
        subgraph_data = graph.x[subset].clone()

        # calculate new index of center node
        new_index = torch.nonzero(subset == idx).squeeze()

        # set center node feature to 0
        subgraph_data[new_index] = 0

        # create mask for the center node, to calculate the loss only on the center node
        mask = torch.ones(subgraph_data.shape[0], dtype=torch.bool)
        mask[new_index] = False

        return Data(x=subgraph_data, edge_index=edge_index, y=graph.x[idx].view(1, 550), mask=mask)

        #get subgraph
        subset, edge_index, mapping, edge_mask =  k_hop_subgraph(node_idx=[idx], edge_index=graph.edge_index, num_hops=1, relabel_nodes=True)
        return Data(x=graph.x[subset], edge_index=edge_index)




class InMemoryGraphDataset(InMemoryDataset):

    def __init__(self, root,data_names, *args, **kwargs):
        self.data_names = data_names
        #self.data_loader = FullImageConstracter(*args, **kwargs)
        self.root = root
        super(InMemoryDataset, self).__init__(root, transform=None, pre_transform=None)


    @property
    def raw_file_names(self):
        return []  # Not used as data is loaded in-memory from `__init__`.

    @property
    def processed_file_names(self):
        return [f"{self.data_names}.pt"]

    def download(self):
        pass  # Not used as data is loaded in-memory from `__init__`.

    def process(self):
        self.data_loader.load_data()
        data_list = self.data_loader.construct_graph()  # Here you generate your data
        data, slices = self.collate(data_list)
        torch.save((data, slices), os.path.join(self.root, self.data_names + ".pt"))


