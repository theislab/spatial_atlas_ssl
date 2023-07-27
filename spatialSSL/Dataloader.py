# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import networkx as nx
import numpy as np
import scanpy as sc
import squidpy as sq
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix


class Dataloader(ABC):
    def __init__(self, file_path: str, image_col: str, label_col: str, include_label: bool, radius: float,
                 node_level: int = 1, batch_size: int = 64, split_percent: tuple = (0.8, 0.1, 0.1)):
        self.file_path = file_path
        self.image_col = image_col
        self.label_col = label_col
        self.node_level = node_level
        self.include_label = include_label
        self.radius = radius
        self.batch_size = batch_size
        self.split_percent = split_percent

        self.adata = None

    def load_data(self):
        # Load data from .h5ad file and return a scanpy AnnData object
        self.adata = sc.read(self.file_path)

    @abstractmethod
    def construct_graph(self):
        pass

    @abstractmethod
    def split_data(self, loader):
        pass

    def build_graph(self):
        pass


class EgoNetDataloader(Dataloader):
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
        super(EgoNetDataloader, self).__init__(*args, **kwargs)

    def construct_graph(self):
        # Constructing graph from coordinates using scanpy's spatial_neighbors function
        images = np.unique(self.adata.obs[self.image_col])
        print(len(images), "images found.")

        sub_g_ensemble = []

        for image in tqdm(images, desc=f"Processing {len(images)} Images"):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image].copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            # Create subgraphs for each node
            g = nx.Graph()

            # Add nodes with features to the graph
            for i, features in enumerate(self.adata.X.toarray()):
                g.add_node(i, features=features)

            # Add edges to the graph
            # print('Adding edges...')
            g.add_edges_from(edge_index.t().tolist())

            # Create subgraphs for each node of g
            subgraphs = [nx.ego_graph(g, node, radius=self.node_level) for node in
                                  tqdm(g.nodes(), desc="Creating Subgraphs", leave=False)]

            # Convert networkx graphs to PyG format
            sub_g_dataset = [from_networkx(graph, group_node_attrs=['features']) for graph in
                             tqdm(subgraphs, desc="Converting to PyG format", leave=False)]

            # Extend the ensemble with the new subgraphs
            sub_g_ensemble.extend(sub_g_dataset)

        loader = DataLoader(sub_g_ensemble, batch_size=self.batch_size, shuffle=True)
        return loader

    def split_data(self, loader):
        # Assuming split_percent is a tuple like (0.7, 0.2, 0.1)
        train_size = int(self.split_percent[0] * len(loader.dataset))
        val_size = int(self.split_percent[1] * len(loader.dataset))
        test_size = len(loader.dataset) - train_size - val_size

        print(train_size, val_size, test_size)

        train_data, val_data, test_data = random_split(loader.dataset, [train_size, val_size, test_size])

        # Create data loaders for each set
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader


class FullImageDataloader(Dataloader):
    def __init__(self, *args, **kwargs):  # TODO: Add default parameters for this methods (e.g. batch_size=4)...
        super(FullImageDataloader, self).__init__(*args, **kwargs)

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

    def split_data(self, graph_dict):
        # split by entire images
        images = list(graph_dict.keys())

        # TODO: Add random state global variable, use different method for splitting data (random split)
        train_images, test_images = train_test_split(images, test_size=(self.split_percent[1] + self.split_percent[2]),
                                                     random_state=42)
        val_images, test_images = train_test_split(test_images, test_size=self.split_percent[2] / (
                self.split_percent[1] + self.split_percent[2]), random_state=42)

        train_data = [graph_dict[image] for image in train_images]
        val_data = [graph_dict[image] for image in val_images]
        test_data = [graph_dict[image] for image in test_images]

        # Create data loaders for each set
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
