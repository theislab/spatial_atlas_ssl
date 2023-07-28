# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import networkx as nx
import numpy as np
import scanpy as sc
import squidpy as sq


from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
from tqdm.auto import tqdm
from spatialSSL.Dataset import EgoNetDataset, FullImageDataset


class SpatialDataloader(ABC):
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

        self.dataset = None
        self.adata = None

    def load_data(self):
        # Load data from .h5ad file and return a scanpy AnnData object
        self.adata = sc.read(self.file_path)

    @abstractmethod
    def construct_graph(self):
        pass



class EgoNetDataloader(SpatialDataloader):
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

        graphs = []

        for image in tqdm(images, desc=f"Processing {len(images)} Images"):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image].copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            graphs.append(Data(x=sub_adata.X.toarray(), edge_index=edge_index))

            """
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
            """
        self.dataset = EgoNetDataset(graphs)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return loader


class FullImageDataloader(SpatialDataloader):
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
            ''' 
            # Construct graph
            g = nx.Graph()
            # Adding nodes
            for i, features in enumerate(sub_adata.X.toarray()):
                g.add_node(i, features=features)
            # Adding edges
            g.add_edges_from(edge_index.t().tolist())

            # Convert networkx graph to PyG format
            graph = from_networkx(g)
            ''' 
            
            graph = Data(x=sub_adata.X.toarray(), edge_index=edge_index)
            graph_dict[image] = graph
            
        self.dataset = FullImageDataset(graph_dict)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        return loader
