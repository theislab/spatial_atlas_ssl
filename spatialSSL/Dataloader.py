# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import numpy as np
import scanpy as sc
import squidpy as sq
import torch

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix, k_hop_subgraph
from tqdm.auto import tqdm

from spatialSSL.Dataset import EgoNetDataset


class SpatialDatasetConstructor(ABC):
    def __init__(self, file_path: str, image_col: str, label_col: str, include_label: bool, radius: float,
                 node_level: int = 1, batch_size: int = 64):
        self.file_path = file_path
        self.image_col = image_col
        self.label_col = label_col
        self.node_level = node_level
        self.include_label = include_label
        self.radius = radius
        self.batch_size = batch_size

        self.dataset = None
        self.adata = None

    def load_data(self):
        # Load data from .h5ad file and return a scanpy AnnData object
        self.adata = sc.read(self.file_path)
        """import os
        import psutil
        pid = os.getpid()
        python_process = psutil.Process(pid)
        memoryUse = python_process.memory_info()[0] / 2. ** 30  # memory use in GB...I think
        print(f'memory use: {memoryUse:.2f} GB'.format(**locals()))
        """

    @abstractmethod
    def construct_graph(self):
        pass


class EgoNetDatasetConstructor(SpatialDatasetConstructor):
    def __init__(self, *args, **kwargs):
        """
        Initializes the Ego_net_dataloader.


        Parameters:
            file_path (str): Path to the .h5ad file.
            image_col (str): Column name of the image id.
            label_col (str): Column name of the label.
            include_label (bool): Whether to include the label in the graph.
            radius (float): Radius of the ego graph.
            node_level (int): Number of node levels to include in the ego graph.
            batch_size (int): Batch size for the data loader.
            split_percent (tuple): Tuple of percentages for train, validation, and test sets.

        This loader creates one subgraph for each node in the dataset by using all nodes within "node_level"
        distance of the center node.
        """
        # TODO: Add default parameters for this methods (e.g. batch_size=32)
        super(EgoNetDatasetConstructor, self).__init__(*args, **kwargs)

    def construct_graph(self):
        # Constructing graph from coordinates using scanpy's spatial_neighbors function
        images = np.unique(self.adata.obs[self.image_col])

        graphs = []

        for image in tqdm(images, desc=f"Processing {len(images)} images"):

            # subset adata to only include cells from the current image
            sub_adata = self.adata[self.adata.obs[self.image_col] == image]#.copy()

            # calculate graph using neighbors function
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")

            edge_index_full, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            # convert to pytorch tensor
            x = torch.tensor(sub_adata.X.toarray(), dtype=torch.double)

            for idx in tqdm(range(len(sub_adata)), desc=f"Processing {len(sub_adata)} nodes", leave=False):
                # create subgraph for each node

                try:
                    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=[idx], edge_index=edge_index_full,
                                                                            num_hops=self.node_level, relabel_nodes=True)
                except IndexError:
                    break

                # skill if subgraph is empty
                if edge_index.shape[1] == 0:
                    continue

                subgraph_data = x[subset].clone()

                # calculate new index of center node
                new_index = torch.nonzero(subset == idx).squeeze()

                # set center node feature to 0
                subgraph_data[new_index] = 0

                # create mask for the center node, to calculate the loss only on the center node
                mask = torch.ones(subgraph_data.shape[0], dtype=torch.bool)
                mask[new_index] = False

                graphs.append(
                    Data(x=subgraph_data, y=x[idx].view(1, 550), edge_index=edge_index, image=image, mask=mask))



            # remove cells from current image from adata
            self.adata = self.adata[self.adata.obs[self.image_col] != image]

        return graphs
        # graphs.append(Data(x=x, edge_index=edge_index, image=image))
        #if self.include_label:
        #                 y = torch.tensor(sub_adata.obs[self.label_col][mapping], dtype=torch.long)
        #                 graphs.append(Data(x=x[subset], edge_index=edge_index, y=y, image=image))
        #             else:
        # Create dataset from graphs
        # self.dataset = EgoNetDataset(graphs=graphs, num_hops=self.node_level)
        # loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # print("total number of cell/nodes: ", len(self.dataset))
        # return loader


class FullImageDatasetConstructor(SpatialDatasetConstructor):
    def __init__(self, *args, **kwargs):  # TODO: Add default parameters for this methods (e.g. batch_size=4)...
        super(FullImageDatasetConstructor, self).__init__(*args, **kwargs)

    def construct_graph(self):
        # Constructing graph from coordinates using scanpy's spatial_neighbors function
        images = np.unique(self.adata.obs[self.image_col])

        graph_dict = {}
        for image in tqdm(images, desc="Constructing Graphs"):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image].copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            # Construct graph
            g = nx.Graph()
            # Adding nodes
            for i, features in enumerate(sub_adata.X.toarray()):
                g.add_node(i, features=features)
            # Adding edges
            g.add_edges_from(edge_index.t().tolist())

            # Convert networkx graph to PyG format
            graph = from_networkx(g)
            graph_dict[image] = graph

        return graph_dict
