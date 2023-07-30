# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import networkx as nx
import numpy as np
import scanpy as sc
import squidpy as sq

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix, k_hop_subgraph
from tqdm.auto import tqdm
from spatialSSL.Dataset import EgoNetDataset, FullImageDataset
import torch


class SpatialDataloader(ABC):
    def __init__(self, file_path: str, image_col: str, label_col: str, include_label: bool, radius: float,
                 node_level: int = 1, batch_size: int = 64, split_percent: tuple = (0.8, 0.1, 0.1), masked_method: str = 'random' , *args, **kwargs):
        self.file_path = file_path
        self.image_col = image_col
        self.label_col = label_col
        self.node_level = node_level
        self.include_label = include_label
        self.radius = radius
        self.batch_size = batch_size
        self.split_percent = split_percent
        self.masked_method = masked_method
        self.dataset = None
        self.adata = None
        self.celltype_to_mask = None

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


class FullImageConstracter(SpatialDataloader):
    def __init__(self, *args, **kwargs):
        super(FullImageConstracter, self).__init__(*args, **kwargs)

    def construct_graph(self):
        images = np.unique(self.adata.obs[self.image_col])

        graphs = []
        for image in tqdm(images, desc="Constructing Graphs"):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image].copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            cell_type = sub_adata.obs[self.label_col].values

            # assuming gene expression is stored in sub_adata.X
            gene_expression = sub_adata.X.toarray()

            # select masking technique
            if self.masked_method == 'random':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_random(gene_expression, cell_type)
            elif self.masked_method == 'cell_type':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_by_cell_type(gene_expression, cell_type, cell_type_to_mask=self.celltype_to_mask)
            elif self.masked_method == 'niche':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_by_niche(gene_expression, cell_type, edge_index)

            # create a mask of size equal to the number of cells
            gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_random(gene_expression, cell_type)

            gene_expression = torch.tensor(gene_expression, dtype=torch.double)
            gene_expression_masked = torch.tensor(gene_expression_masked, dtype=torch.double)
            print(cell_type.shape)
            graph = Data(x=gene_expression, edge_index=edge_index, y=gene_expression_masked, mask=mask,
                         cell_type=cell_type, cell_type_masked=cell_type_masked, image=image)
            graphs.append(graph)
        return graphs

    @staticmethod
    def masking_random(gene_expression, cell_type):
        # create a mask of size equal to the number of cells

        # Mask is ture for cells that are not masked
        mask = torch.ones(gene_expression.shape[0], dtype=torch.bool)

        # randomly select some percentage of cells to mask
        num_cells_to_mask = int(gene_expression.shape[0] * 0.2)  # e.g., 10%
        cells_to_mask = np.random.choice(gene_expression.shape[0], size=num_cells_to_mask, replace=False)
        mask[cells_to_mask] = False

        # save the masked gene expression
        gene_expression_masked = gene_expression[~mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[~cells_to_mask]

        return gene_expression, gene_expression_masked, mask, cell_type_masked

    @staticmethod
    def masking_by_cell_type(gene_expression, cell_type, cell_type_to_mask):

        # Create a mask of size equal to the number of a cell type
        # Mask is ture for cells that are not masked
        mask = torch.ones(gene_expression.shape[0], dtype=torch.bool)
        cells_to_mask = np.where(cell_type == cell_type_to_mask)[0]
        mask[cells_to_mask] = False

        # save the masked gene expression
        gene_expression_masked = gene_expression[~mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[~cells_to_mask]

        return gene_expression, gene_expression_masked, mask, cell_type_masked

    @staticmethod
    def masking_by_niche(gene_expression, cell_type, edge_index, extend=1):
        # random select few cells then get the neighbors of those cells
        num_cells_to_mask = int(gene_expression.shape[0] * 0.1)  # e.g., 10%
        cells_to_mask = np.random.choice(gene_expression.shape[0], size=num_cells_to_mask, replace=False)
        cells_to_mask_neighbors = []

        # get the neighbors of the cells to mask with degree
        for cell in cells_to_mask:
            subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=cell, edge_index=edge_index,
                                                                    num_hops=extend, relabel_nodes=False)
            cells_to_mask_neighbors.extend(subset)

        cells_to_mask_neighbors = np.unique(cells_to_mask_neighbors)
        cells_to_mask = np.concatenate((cells_to_mask, cells_to_mask_neighbors))

        # Create a mask of size equal to the number of a cell type
        # Mask is ture for cells that are not masked
        mask = torch.ones(gene_expression.shape[0], dtype=torch.bool)
        mask[cells_to_mask] = False

        # save the masked gene expression
        gene_expression_masked = gene_expression[~mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[~mask]

        return gene_expression, gene_expression_masked, mask, cell_type_masked


