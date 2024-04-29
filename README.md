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

1. **K-Nearest Neighbors (KNN)**
   - **Methodology**: KNN is a non-parametric, instance-based learning algorithm. It predicts the class of a sample by finding the majority class among its K nearest neighbors in the feature space. The distance metric, often Euclidean distance, is used to measure the similarity between instances.
   - **Parameters Tuned**: The main parameter to tune is the number of neighbors, K.
   - **Implementation**: `models/knn.py`

2. **Mini Batch KMeans**
   - **Methodology**: Mini Batch KMeans is a variant of KMeans that uses mini-batches of data to update cluster centroids, making it faster and more scalable for large datasets. It performs KMeans clustering on small random subsets of the data, updating cluster centroids based on these mini-batches.
   - **Parameters Tuned**: Parameters include the number of clusters (k) and the batch size.
   - **Implementation**: `models/mini_batch_kmeans.py`

3. **Hierarchical Clustering**
   - **Methodology**: Hierarchical clustering builds a hierarchy of clusters by recursively merging or splitting clusters based on a distance metric. It does not require the number of clusters to be specified in advance and produces a dendrogram that can be cut at different levels to obtain clusters.
   - **Parameters Tuned**: Parameters include the linkage criterion and the distance metric.
   - **Implementation**: `models/hierarchical_clustering.py`

4. **Birch**
   - **Methodology**: Birch (Balanced Iterative Reducing and Clustering using Hierarchies) is a hierarchical clustering algorithm designed for large datasets. It incrementally clusters the data by building a tree structure called the Clustering Feature Tree (CF Tree) to efficiently represent the data distribution.
   - **Parameters Tuned**: Parameters include the branching factor, threshold, and the clustering criterion.
   - **Implementation**: `models/birch.py`

5. **Gaussian Mixture Models (GMM)**
   - **Methodology**: Gaussian Mixture Models represent the distribution of data as a mixture of several Gaussian distributions. GMM clustering assigns data points to clusters by maximizing the likelihood that the data is generated from a mixture of Gaussians.
   - **Parameters Tuned**: Parameters include the number of components and the covariance type.
   - **Implementation**: `models/gmm.py`

6. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
   - **Methodology**: DBSCAN is a density-based clustering algorithm that groups together closely packed points based on a distance measure and a minimum number of points within the neighborhood of a data point. It is capable of discovering clusters of arbitrary shapes and sizes and can identify outliers as noise.
   - **Parameters Tuned**: Parameters include the epsilon (maximum distance between points in a cluster) and the minimum number of points required to form a dense region (minPts).
   - **Implementation**: `models/dbscan.py`

## Clustering Evaluation Metrics

### Fowlkes-Mallows Score

The Fowlkes-Mallows score is a metric used to evaluate the similarity of two clusterings. It computes the similarity between the true labels and the predicted clusters. It ranges from 0 to 1, where 1 indicates perfect similarity between the two clusterings.

### Silhouette Score

The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

### Adjusted Rand Score

The adjusted Rand score is a measure of the similarity between two clusterings, adjusted for chance. It considers all pairs of samples and counts pairs that are assigned in the same or different clusters in both the true and predicted clusterings. It ranges from -1 to 1, where 1 indicates perfect similarity between the two clusterings.

### Normalized Mutual Information Score

The normalized mutual information score is a measure of the mutual dependence between the true labels and the predicted clusters, adjusted for chance. It ranges from 0 to 1, where 1 indicates perfect agreement between the true and predicted clusterings.


## Detailed Description
- `main.py`: This script loads the dataset, extracts article embeddings, performs clustering using various algorithms, saves the cluster labels, and computes clustering metrics.
- `utils.py`: This module provides utility functions for handling embeddings, including extraction, saving, loading, and printing clustering metrics such as Fowlkes Mallows Score, Silhouette Score, Adjusted Rand Score, and Normalized Mutual Information (NMI).
- `embeddings_tsne.py`: This script loads the saved embeddings and cluster labels, applies t-SNE to reduce the dimensionality of the embeddings to 2D, and visualizes the embeddings with colored clusters.

## Acknowledgments
- This project utilizes the Sentence Transformers library for generating embeddings from articles.
- The t-SNE visualization is implemented using the scikit-learn library.

Finally there is a medium article you can read for deeper insight. This is a [link](https://medium.com/@a.r.amouzad.m/classic-machine-learning-part-4-4-clustering-on-text-dataset-e12520edd2f0) to story.