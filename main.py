import pandas as pd
from utils import load_concatenated_embeddings, extract_and_concatenate_embeddings, save_concatenated_embeddings, print_cluster_metrics

from models.kmeans import cluster_embeddings_kmeans
from models.hierarchical import cluster_embeddings_hierarchical
from models.birch import cluster_embeddings_birch
# from models.dbscan import grid_search_dbscan, cluster_embeddings_dbscan
from models.gmm import cluster_embeddings_gmm
from models.mini_batch_kmeans import cluster_embeddings_mini_batch_kmeans

def main():
    # Load dataset from CSV
    dataset_path = 'dataset/BBC_News_Train.csv'
    df = pd.read_csv(dataset_path)

    # Extract articles and categories
    articles = df['Text'].tolist()
    categories = df['Category'].unique()
    labels_map = {category: i for i, category in enumerate(categories)}
    labels = df['Category'].map(labels_map).values

    # Check if concatenated embeddings are already saved
    saved_embeddings = load_concatenated_embeddings()
    if saved_embeddings is None:
        # Extract and concatenate embeddings
        concatenated_embeddings = extract_and_concatenate_embeddings(articles)

        # Save concatenated embeddings
        save_concatenated_embeddings(concatenated_embeddings)
    else:
        concatenated_embeddings = saved_embeddings

    # Cluster embeddings using KMeans
    kmeans_results = cluster_embeddings_kmeans(concatenated_embeddings, labels, n_clusters=5)
    print_cluster_metrics("KMeans", *kmeans_results[1:])

    # Cluster embeddings using Mini Batch KMeans
    mini_batch_kmeans_results = cluster_embeddings_mini_batch_kmeans(concatenated_embeddings, labels, n_clusters=5)
    print_cluster_metrics("Mini Batch KMeans", *mini_batch_kmeans_results[1:])

    # Cluster embeddings using Hierarchical Clustering
    hierarchical_results = cluster_embeddings_hierarchical(concatenated_embeddings, labels, n_clusters=5)
    print_cluster_metrics("Hierarchical Clustering", *hierarchical_results[1:])

    # Cluster embeddings using Birch
    birch_results = cluster_embeddings_hierarchical(concatenated_embeddings, labels, n_clusters=5)
    print_cluster_metrics("Birch", *birch_results[1:])

    # Grid search for best DBSCAN parameters
    # best_params = grid_search_dbscan(concatenated_embeddings, labels)
    # print("Best DBSCAN parameters:", best_params)

    # Cluster embeddings using DBSCAN with best parameters
    # dbscan_results = cluster_embeddings_dbscan(concatenated_embeddings, labels, eps=best_params['eps'], min_samples=best_params['min_samples'])
    # print_cluster_metrics("DBSCAN", *dbscan_results[1:])

    # Cluster embeddings using Gaussian Mixture Models (GMM)
    gmm_results = cluster_embeddings_gmm(concatenated_embeddings, labels, n_components=5)
    print_cluster_metrics("Gaussian Mixture Models", *gmm_results[1:])

    # Create a DataFrame to store the cluster labels
    results_df = pd.DataFrame({
        'Real_Labels' : labels,
        'KMeans_Labels': kmeans_results[0],
        'Mini_Batch_KMeans' : mini_batch_kmeans_results[0],
        'Hierarchical_Labels': hierarchical_results[0],
        'Birch_Labels': birch_results[0],
        'GMM_Labels': gmm_results[0],
    })

    results_df.to_csv('result_labels.csv')

if __name__ == "__main__":
    main()
