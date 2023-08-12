# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
from tqdm.auto import tqdm


class SpatialDatasetConstructor(ABC):
    def __init__(self, file_path: str, image_col: str, label_col: str, include_label: bool, radius: float,
                 node_level: int = 1, mask_method='random', **kwargs):
        self.file_path = file_path
        self.image_col = image_col
        self.label_col = label_col
        self.node_level = node_level
        self.include_label = include_label
        self.radius = radius

        self.mask_method = mask_method
        self.dataset = None
        self.adata = None
        if self.mask_method == 'cell_type':
            self.celltype_to_mask = kwargs['celltype_to_mask']

    def load_data(self):
        # Load data from .h5ad file and return a scanpy AnnData object
        self.adata = sc.read(self.file_path)

        # Extract the column values to a pandas DataFrame
        obs_df = self.adata.obs.copy()

        # Sort the DataFrame by the specified column
        sorted_obs_df = obs_df.sort_values(by=self.image_col)

        # Reindex the AnnData object using the sorted index of the DataFrame
        self.adata = self.adata[sorted_obs_df.index]


        # self.adata.X = torch.tensor(self.adata.X., dtype=torch.double)
        # Create a dictionary of AnnData objects, one for each image
        # self.adatas = {image_id: adata[adata.obs[self.image_col] == image_id] for image_id in
        #               np.unique(adata.obs[self.image_col])}

        # del adata
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

    def construct_graph(self, show_progress_bar=False):

        # Constructing graph_info from coordinates using scanpy's spatial_neighbors function
        images = pd.unique(self.adata.obs[self.image_col])

        graphs = {}

        index = 0

        for image in tqdm(images, desc=f"Processing {len(images)} images", disable=not show_progress_bar):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image]  # .copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index_full, edge_weight_full = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_distances'])
            #edge_index_full, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])
            graphs[image] = (edge_index_full, index, edge_weight_full)
            index += len(sub_adata)


        subgraphs = []

        # for image in tqdm(images, desc=f"Processing {len(images)} images"):
        for idx in tqdm(range(len(self.adata)), desc=f"Processing {len(self.adata)} nodes", disable=not show_progress_bar):
            # create subgraph for each node
            try:


                graph_info = graphs[self.adata.obs[self.image_col][idx]]
                offset = graph_info[1]


                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=[idx - offset], edge_index=
                graph_info[0], num_hops=self.node_level, relabel_nodes=True)

                if len(subset) == 1:
                    continue

                new_index = torch.nonzero(subset == idx - offset).squeeze()
                mask = torch.zeros(subset.shape[0], dtype=torch.bool)
                mask[new_index] = True

                data = Data(x=subset + offset, y=idx, edge_index=edge_index, mask=mask, edge_weights=graph_info[2][edge_mask])#, celltype=self.adata.obs[self.label_col][idx])
                subgraphs.append(data)
            except Exception as e:
                print(e)
                print(f"Error processing node {idx}")
                continue

        return subgraphs



class FullImageDatasetConstructor(SpatialDatasetConstructor):
    def __init__(self, *args, **kwargs):  # TODO: Add default parameters for this methods (e.g. batch_size=4)...
        super(FullImageDatasetConstructor, self).__init__(*args, **kwargs)

    def construct_graph(self):
        # Constructing graph from coordinates using scanpy's spatial_neighbors function
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
            if self.mask_method == 'random':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_random(gene_expression,
                                                                                                      cell_type)
            elif self.mask_method == 'cell_type':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_by_cell_type(
                    gene_expression, cell_type, cell_type_to_mask=self.celltype_to_mask)
            elif self.mask_method == 'niche':
                gene_expression, gene_expression_masked, mask, cell_type_masked = self.masking_by_niche(gene_expression,
                                                                                                        cell_type,
                                                                                                        edge_index)

            # convert to tensors
            gene_expression = torch.tensor(gene_expression, dtype=torch.double)
            gene_expression_masked = torch.tensor(gene_expression_masked, dtype=torch.double)
            # print(cell_type.shape)
            graph = Data(x=gene_expression, edge_index=edge_index, y=gene_expression_masked, mask=mask,
                         cell_type=cell_type, cell_type_masked=cell_type_masked, image=image)
            graphs.append(graph)
        return graphs

    @staticmethod
    def masking_random(gene_expression, cell_type):
        # create a mask of size equal to the number of cells

        # Mask is ture for cells that are masked
        mask = torch.zeros(gene_expression.shape[0], dtype=torch.bool)

        # randomly select some percentage of cells to mask
        num_cells_to_mask = int(gene_expression.shape[0] * 0.2)  # e.g., 10%
        cells_to_mask = np.random.choice(gene_expression.shape[0], size=num_cells_to_mask, replace=False)
        mask[cells_to_mask] = True

        # save the masked gene expression
        gene_expression_masked = gene_expression[mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[cells_to_mask]

        return gene_expression, gene_expression_masked, mask, cell_type_masked

    @staticmethod
    def masking_by_cell_type(gene_expression, cell_type, cell_type_to_mask):

        # Create a mask of size equal to the number of a cell type
        # Mask is ture for cells that are masked
        mask = torch.zeros(gene_expression.shape[0], dtype=torch.bool)

        # check if cell type to mask is in the dataset
        if cell_type_to_mask not in np.unique(cell_type):
            raise ValueError(f"Cell type {cell_type_to_mask} not in the dataset")

        # get the cells of the cell type to mask
        cells_to_mask = np.where(cell_type == cell_type_to_mask)[0]

        mask[cells_to_mask] = True
        # Count the number of True values in the mask
        # um_true = sum(mask)

        # print("Number of True values in the mask:", num_true)
        # save the masked gene expression
        gene_expression_masked = gene_expression[mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[cells_to_mask]

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
        mask = torch.zeros(gene_expression.shape[0], dtype=torch.bool)
        mask[cells_to_mask] = True

        # save the masked gene expression
        gene_expression_masked = gene_expression[mask]

        # set the gene expression of the masked cells to zero
        gene_expression[cells_to_mask] = 0

        # keep track of the cell types of the masked cells
        cell_type_masked = cell_type[cells_to_mask]

        return gene_expression, gene_expression_masked, mask, cell_type_masked
