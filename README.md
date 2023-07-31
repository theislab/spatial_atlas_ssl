# SpatialSSL: Reconstruction and Generalization of Whole-Brain Spatial Transcriptomics in Mouse Brain with Self-Supervised Learning

## Course: Masterpraktikum

### Team

- Leopold Endres
- Cheng Wei

### Supervisors

- Till Richter
- Anna Schaar
- Francesca Drummer

## Introduction

SpatialSSL is a research project that explores different methods of self-supervised learning applied to spatial
transcriptomics data in the mouse brain. Self-supervised learning is a type of machine learning where models are trained
to predict certain aspects of the data without explicit human annotations. The project focuses on modeling spatial data
as graphs and applying graph-based machine learning and pretraining methods. The goal of this project is to analyse how
different pretrained models perform on downstream tasks.

### Provided Data

The dataset used in SpatialSSL leverages the BICCN 2.0 dataset, which contains spatial data of approximately 4 million
brain cells, measuring the expression of 550 genes. Spatial transcriptomics data combines positional information (
usually in the form of x-y coordinates) with gene expression for each cell. The dataset is organized into 59 "images,"
each containing spatially independent cells. Depending on the annotation levels, there are 15, 30, or 35 different cell
types present.

### Preprocessing

Placeholder for information about the preprocessing steps.

### Dataset Construction - Two Methods

1. **Create Single Graph for Each Image in the Dataset:**
    - Parameters: Radius, threshold distance between cells to draw edges between them.
    - This method results in relatively large graphs spanning from 30k to 60k nodes.

2. **Splicing Graphs into Subgraphs Using Egonet of Each Node:**
    - Parameters: Radius, threshold distance between cells to draw edges between them; k_hop: the number of "hops" for
      subgraph creation.
    - This method results in approximately 4 million small graphs, with the size depending on the radius and k_hop
      parameters.

### Masking Techniques for Pretraining

For both datasets, several masking methods were explored during the pretraining phase:

- Masking all gene expression values of 20% of cells.
- Masking specific cell types.

**Specific to EgoNetDataset:**

- Masking the gene expression of the center node (the node from which the subgraph was created).

### Pretraining

Pretraining of the models was performed using graph neural networks with graph attention and convolution.

### Downstream Tasks

SpatialSSL builds different models for a set of downstream tasks. The pretrained models' weights are loaded and frozen,
and additional layers are added for fine-tuning. Two downstream tasks were benchmarked using pretrained models and
randomly initialized models:

1. **Gene Expression Inpainting of Masked Cells:**
    - The model is tasked with predicting the gene expression values for masked cells.

2. **Cell Type Prediction:**
    - The model is trained to predict the cell type of each brain cell.

## How to Use This Repository

Placeholder for instructions on how to use the code and reproduce the experiments.