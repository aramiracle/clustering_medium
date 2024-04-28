# Article Clustering Project

## Overview
This project focuses on clustering articles from the BBC News dataset into distinct categories using various clustering algorithms and visualizing the embeddings in 2D space using t-SNE (t-distributed Stochastic Neighbor Embedding). The clustering algorithms employed include KMeans, Mini Batch KMeans, Hierarchical Clustering, Birch, and Gaussian Mixture Models (GMM). This README provides an overview of the project structure, usage instructions, requirements, clustering methods, and acknowledgments.

## File Structure
- `main.py`: Python script responsible for clustering articles using different algorithms and saving the cluster labels.
- `utils.py`: Contains utility functions for extracting and saving embeddings, loading embeddings, and printing clustering metrics.
- `embeddings_tsne.py`: Python script for visualizing embeddings in 2D space using t-SNE.

## Usage
1. **Setup Environment**: Ensure you have the required Python libraries installed. You can install them via pip using `pip install -r requirements.txt`.
2. **Prepare Dataset**: The dataset used in this project is the BBC News dataset. You can download it from [Kaggle](https://www.kaggle.com/c/learn-ai-bbc). Place the dataset file (`BBC_News_Train.csv`) in the `dataset` directory. The dataset should be in CSV format with columns `Text` containing the articles and `Category` containing the category labels.
3. **Run Clustering**: Execute `main.py` to cluster the articles using the specified algorithms. The clustering results will be saved in `results_labels.csv`.
4. **Visualize Embeddings**: After clustering, run `embeddings_tsne.py` to visualize the embeddings in 2D space using t-SNE. The visualization will be saved as `embedding_with_clusters.png`.

## Requirements
Ensure you have the following software and libraries installed:
- Python 3.x
- Pandas
- tqdm
- torch
- numpy
- scikit-learn
- sentence-transformers
- matplotlib

You can install these dependencies using the command `pip install -r requirements.txt`.

## Clustering Methods
### 1. KMeans
- KMeans is a popular clustering algorithm that partitions data into k clusters by minimizing the within-cluster variance.
- It works by iteratively assigning each data point to the nearest cluster centroid and updating the centroids based on the mean of the points assigned to each cluster.

### 2. Mini Batch KMeans
- Mini Batch KMeans is a variant of KMeans that uses mini-batches of data to update cluster centroids, making it faster and more scalable for large datasets.
- It performs KMeans clustering on small random subsets of the data, updating cluster centroids based on these mini-batches.

### 3. Hierarchical Clustering
- Hierarchical clustering builds a hierarchy of clusters by recursively merging or splitting clusters based on a distance metric.
- It does not require the number of clusters to be specified in advance and produces a dendrogram that can be cut at different levels to obtain clusters.

### 4. Birch
- Birch (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm designed for large datasets.
- It incrementally clusters the data by building a tree structure called the Clustering Feature Tree (CF Tree) to efficiently represent the data distribution.

### 5. Gaussian Mixture Models (GMM)
- Gaussian Mixture Models represent the distribution of data as a mixture of several Gaussian distributions.
- GMM clustering assigns data points to clusters by maximizing the likelihood that the data is generated from a mixture of Gaussians.

## Detailed Description
- `main.py`: This script loads the dataset, extracts article embeddings, performs clustering using various algorithms, saves the cluster labels, and computes clustering metrics.
- `utils.py`: This module provides utility functions for handling embeddings, including extraction, saving, loading, and printing clustering metrics such as Fowlkes Mallows Score, Silhouette Score, Adjusted Rand Score, and Normalized Mutual Information (NMI).
- `embeddings_tsne.py`: This script loads the saved embeddings and cluster labels, applies t-SNE to reduce the dimensionality of the embeddings to 2D, and visualizes the embeddings with colored clusters.

## Acknowledgments
- This project utilizes the Sentence Transformers library for generating embeddings from articles.
- The t-SNE visualization is implemented using the scikit-learn library.