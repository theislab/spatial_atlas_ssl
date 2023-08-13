# Standard Library Imports
from abc import ABC, abstractmethod

# Third-Party Library Imports
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, k_hop_subgraph
from tqdm.auto import tqdm


class SpatialDatasetConstructor(ABC):
    def __init__(self, file_path: str, image_col: str, label_col: str, include_label: bool, radius: float,
                 node_level: int = 1,
                 **kwargs):
        self.file_path = file_path
        self.image_col = image_col
        self.label_col = label_col
        self.node_level = node_level
        self.include_label = include_label
        self.radius = radius
        self.dataset = None
        self.adata = None

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

    def construct_graph(self):

        # Constructing graph from coordinates using scanpy's spatial_neighbors function
        images = pd.unique(self.adata.obs[self.image_col])

        graphs = {}

        index = 0

        for image in tqdm(images, desc=f"Processing {len(images)} images"):
            sub_adata = self.adata[self.adata.obs[self.image_col] == image]  # .copy()
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")
            edge_index_full, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])
            graphs[image] = (edge_index_full, index)
            index += len(sub_adata)

            # graphs.append(edge_index_full)

        subgraphs = []

        # for image in tqdm(images, desc=f"Processing {len(images)} images"):
        for idx in tqdm(range(len(self.adata)), desc=f"Processing {len(self.adata)} nodes"):
            # create subgraph for each node
            try:

                offset = graphs[self.adata.obs[self.image_col][idx]][1]

                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=[idx - offset], edge_index=
                graphs[self.adata.obs[self.image_col][idx]][0],
                                                                        num_hops=self.node_level, relabel_nodes=True)
                # convert to pytorch tensor
                # x = torch.tensor(subset.X.toarray(), dtype=torch.double)
                # y = torch.tensor(subset.obs[self.label_col].values, dtype=torch.long)
                # edge_index = torch.tensor(edge_index, dtype=torch.long)
                # edge_mask = torch.tensor(edge_mask, dtype=torch.bool)
                # mapping = torch.tensor(mapping, dtype=torch.long)
                # create data object
                new_index = torch.nonzero(subset == idx - offset).squeeze()
                mask = torch.zeros(subset.shape[0], dtype=torch.bool)
                mask[new_index] = True
                data = Data(x=subset + offset, y=idx, edge_index=edge_index, mask=mask)
                subgraphs.append(data)
            except Exception as e:
                print(f"Error processing node {idx}")
                continue

        return subgraphs

        # pre calculate the graphs for each image

        for image in tqdm(self.adatas.keys()):  # tqdm(images, desc=f"Processing {len(images)} images"):

            sub_adata = self.adatas[image]
            # subset adata to only include cells from the current image
            # sub_adata = self.adata[self.adata.obs[self.image_col] == image]#.copy()

            # calculate graph using neighbors function
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")

            edge_index_full, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            # convert to pytorch tensor
            # x = torch.tensor(sub_adata.X.toarray(), dtype=torch.double)

            for idx in tqdm(range(len(sub_adata)), desc=f"Processing {len(sub_adata)} nodes", leave=False):
                # create subgraph for each node

                try:
                    subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=[idx], edge_index=edge_index_full,
                                                                            num_hops=self.node_level,
                                                                            relabel_nodes=True)
                except IndexError:
                    continue

                # skip if subgraph is empty, zero edges
                if edge_index.shape[1] == 0:
                    continue

                # subgraph_data = x[subset].clone()

                # calculate new index of center node
                new_index = torch.nonzero(subset == idx).squeeze()

                # set center node feature to 0
                # subgraph_data[new_index] = 0

                # create mask for the center node, to calculate the loss only on the center node
                mask = torch.ones(subset.shape[0], dtype=torch.bool)
                mask[new_index] = False

                graphs.append(
                    Data(x=subset, y=idx, edge_index=edge_index, image=image, mask=mask))
                # graphs.append(
                #   Data(x=subgraph_data, y=x[idx].view(1, 550), edge_index=edge_index, image=image, mask=mask))

            # print(f"number of subgraphs: {len(graphs)}")

            # remove adata from memory
            # self.adatas[image] = None

            # remove cells from current image from adata
            # self.adata = self.adata[self.adata.obs[self.image_col] != image]

        return graphs
        # graphs.append(Data(x=x, edge_index=edge_index, image=image))
        # if self.include_label:
        #                 y = torch.tensor(sub_adata.obs[self.label_col][mapping], dtype=torch.long)
        #                 graphs.append(Data(x=x[subset], edge_index=edge_index, y=y, image=image))
        #             else:
        # Create dataset from graphs
        # self.dataset = EgoNetDataset(graphs=graphs, num_hops=self.node_level)
        # loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        # print("total number of cell/nodes: ", len(self.dataset))
        # return loader


class FullImageDatasetConstructor(SpatialDatasetConstructor):
    def __init__(self, random_mask_percentage=0.1, mask_method="random", *args, **kwargs):
        super(FullImageDatasetConstructor, self).__init__(*args, **kwargs)
        self.random_mask_percentage = random_mask_percentage
        self.mask_method = mask_method
        self.encode_book = None
        self.cell_type_to_mask = None
        self.niche_to_mask = 1

        if self.mask_method == 'cell_type':
            self.cell_type_to_mask = kwargs['cell_type_to_mask']

        if self.mask_method == 'niche':
            self.niche_to_mask = kwargs['niche_to_mask']

    def construct_graph(self):
        # Constructing graph from coordinates using scanpy's spatial_neighbors function

        cell_mask, cell_mask_index = None, None

        images = np.unique(self.adata.obs[self.image_col])

        encode_cell_type = LabelEncoder()
        encode_cell_type.fit(self.adata.obs[self.label_col].values)
        self.adata.obs['cell_type_encoded'] = encode_cell_type.transform(self.adata.obs[self.label_col].values)

        self.encode_book = dict(zip(encode_cell_type.classes_, encode_cell_type.transform(encode_cell_type.classes_)))

        graphs = []
        for image in tqdm(images, desc="Constructing Graphs"):

            # subset adata to only include cells from the current image
            sub_adata = self.adata[self.adata.obs[self.image_col] == image].copy()

            # calculate graph using neighbors function
            sq.gr.spatial_neighbors(adata=sub_adata, radius=self.radius, key_added="adjacency_matrix",
                                    coord_type="generic")

            edge_index, _ = from_scipy_sparse_matrix(sub_adata.obsp['adjacency_matrix_connectivities'])

            cell_type = sub_adata.obs["cell_type_encoded"].values

            # assuming gene expression is stored in sub_adata.X
            gene_expression_coo = sub_adata.X.tocoo()
            num_cells = gene_expression_coo.shape[0]

            # select masking technique and return graph index for masking

            if self.mask_method == 'random':
                cell_mask, cell_mask_index = self.masking_random(num_cells, self.random_mask_percentage)

            elif self.mask_method == 'cell_type':

                # check if cell type to mask is in the dataset
                if self.cell_type_to_mask not in np.unique(cell_type):
                    print(f"Cell type {self.cell_type_to_mask} not found in image {image}. Skipping this image.")
                    continue

                cell_mask, cell_mask_index = self.masking_by_cell_type(num_cells, cell_type,
                                                                       cell_type_to_mask=self.cell_type_to_mask)

            elif self.mask_method == 'niche':
                cell_mask, cell_mask_index = self.masking_by_niche(num_cells,
                                                                   edge_index,
                                                                   self.niche_to_mask
                                                                   )

            # gene_expression_coo = gene_expression.tocoo()
            # gene_expression_masked_coo = gene_expression_masked.tocoo()

            # convert to pytorch tensor
            # convert to tensors
            gene_expression = torch.sparse_coo_tensor(
                indices=np.vstack((gene_expression_coo.row, gene_expression_coo.col)),
                values=gene_expression_coo.data,
                size=gene_expression_coo.shape,
                dtype=torch.double)

            cell_type = torch.tensor(cell_type)
            cell_mask_index = torch.tensor(cell_mask_index)
            distances = sub_adata.obsp['adjacency_matrix_distances']

            # Get the sparse COO representation of the weights
            weights_coo = distances.tocoo()

            edge_weights_tensor = torch.sparse_coo_tensor(
                indices=np.vstack((weights_coo.row, weights_coo.col)),
                values=weights_coo.data,
                size=weights_coo.shape,
                dtype=torch.double)

            # Include the edge weights in the Data object
            graph = Data(x=gene_expression, edge_index=edge_index, edge_attr=edge_weights_tensor,
                         cell_mask=cell_mask,
                         cell_mask_index=cell_mask_index,
                         cell_type=cell_type, image=image,
                         num_nodes=gene_expression.shape[0])
            graphs.append(graph)

        return graphs

    @staticmethod
    def masking_random(num_cell, random_to_mask=0.1):
        # create a mask of size equal to the number of cells

        # Mask is ture for cells that are masked
        cell_mask = torch.zeros(num_cell, dtype=torch.bool)

        # randomly select some percentage of cells to mask
        num_cells_to_mask = int(num_cell * random_to_mask)
        cell_mask_index = np.random.choice(num_cell, size=num_cells_to_mask, replace=False)
        cell_mask[cell_mask_index] = True

        return cell_mask, cell_mask_index

    @staticmethod
    def masking_by_cell_type(num_cell, cell_type, cell_type_to_mask):

        # Create a mask of size equal to the number of a cell type
        # Mask is ture for cells that are masked
        cell_mask = torch.zeros(num_cell, dtype=torch.bool)

        # get the cells of the cell type to mask
        cell_mask_index = np.where(cell_type == cell_type_to_mask)[0]

        cell_mask[cell_mask_index] = True

        return cell_mask, cell_mask_index

    @staticmethod
    def masking_by_niche(num_cell, edge_index, niche_to_mask=0.05, extend=1):

        # Determine the number of cells to mask based on the type of num_niche
        if isinstance(niche_to_mask, float):
            num_cells_to_mask = int(num_cell * niche_to_mask)
        elif isinstance(niche_to_mask, int):
            num_cells_to_mask = niche_to_mask
        else:
            raise ValueError("num_niche must be either a float or an integer.")

        cell_mask_index = torch.tensor(np.random.choice(num_cell, size=num_cells_to_mask, replace=False))
        cells_to_mask_neighbors = []

        # get the neighbors of the cells to mask with degree
        for cell in tqdm(cell_mask_index):
            subset, _, _, _ = k_hop_subgraph(node_idx=cell.item(), edge_index=edge_index,
                                             num_hops=extend, relabel_nodes=False)  # Change this line
            cells_to_mask_neighbors.extend(subset.cpu().numpy())  # If subset is a tensor, convert to numpy

        cells_to_mask_neighbors = np.unique(cells_to_mask_neighbors)
        cell_mask_index = np.concatenate((cell_mask_index, cells_to_mask_neighbors))

        # Create a mask of size equal to the number of a cell type
        # Mask is ture for cells that are not masked
        cell_mask = torch.zeros(num_cell, dtype=torch.bool)
        cell_mask[cell_mask_index] = True

        return cell_mask, cell_mask_index
